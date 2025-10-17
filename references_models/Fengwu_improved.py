"""
FengWu气象预测模型 - 改进版
基于论文: FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead
https://arxiv.org/pdf/2304.02948.pdf

主要改进:
1. 使用einops重构张量操作，提高代码可读性
2. 复用Pangu Model.py中已改进的模块
3. 添加详细的中文注释
4. 优化代码结构
"""

import math
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat
from collections.abc import Sequence
import warnings

# ==================== 从Pangu Model.py复用的基础模块 ====================

def _trunc_normal_(tensor, mean, std, a, b):
    """截断正态分布初始化"""
    def norm_cdf(x):
        # 计算标准正态累积分布函数
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # 使用截断均匀分布生成值，然后使用逆CDF转换为正态分布
    u1 = norm_cdf((a - mean) / std)
    u2 = norm_cdf((b - mean) / std)

    tensor.uniform_(2 * u1 - 1, 2 * u2 - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """截断正态分布初始化的包装函数"""
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class Mlp(nn.Module):
    """
    多层感知机模块
    包含两个线性层和激活函数，支持dropout
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        # 两层全连接网络，中间使用激活函数和dropout
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """
    Drop Path (随机深度) 实现
    在训练时随机丢弃路径（设置为0），用于正则化
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # 创建与batch size相同的随机mask，其他维度为1
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop Path模块封装"""
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


# ==================== Patch Embedding模块（使用einops改进） ====================

class PatchEmbed2D(nn.Module):
    """
    2D图像到Patch Embedding的转换
    将2D气象场数据划分为patch并进行特征嵌入
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        height, width = img_size
        h_patch_size, w_patch_size = patch_size

        # 计算padding，确保图像尺寸是patch_size的整数倍
        padding_left = padding_right = padding_top = padding_bottom = 0

        h_remainder = height % h_patch_size
        w_remainder = width % w_patch_size

        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = int(h_pad - padding_top)

        if w_remainder:
            w_pad = w_patch_size - w_remainder
            padding_left = w_pad // 2
            padding_right = int(w_pad - padding_left)

        self.pad = nn.ConstantPad2d(
            (padding_left, padding_right, padding_top, padding_bottom), value=0
        )
        # 使用卷积进行patch embedding
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        """
        输入: (B, C, H, W)
        输出: (B, embed_dim, H/patch_h, W/patch_w)
        """
        B, C, H, W = x.shape

        # Padding到patch_size的整数倍
        x = self.pad(x)

        # 卷积进行patch embedding
        x = self.proj(x)

        # 可选的归一化
        if self.norm is not None:
            x = rearrange(x, 'b c h w -> b h w c')
            x = self.norm(x)
            x = rearrange(x, 'b h w c -> b c h w')
        return x


class PatchEmbed3D(nn.Module):
    """
    3D图像到Patch Embedding的转换
    处理包含多个气压层的3D气象数据
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        level, height, width = img_size
        l_patch_size, h_patch_size, w_patch_size = patch_size

        # 计算3D padding
        padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0

        l_remainder = level % l_patch_size
        h_remainder = height % h_patch_size
        w_remainder = width % w_patch_size

        if l_remainder:
            l_pad = l_patch_size - l_remainder
            padding_front = l_pad // 2
            padding_back = l_pad - padding_front
        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = h_pad - padding_top
        if w_remainder:
            w_pad = w_patch_size - w_remainder
            padding_left = w_pad // 2
            padding_right = w_pad - padding_left

        self.pad = nn.ConstantPad3d(
            (padding_left, padding_right, padding_top, padding_bottom,
             padding_front, padding_back), value=0
        )
        # 3D卷积进行patch embedding
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        """
        输入: (B, C, L, H, W) 其中L=气压层
        输出: (B, embed_dim, L/pl, H/ph, W/pw)
        """
        B, C, L, H, W = x.shape

        # Padding到patch_size的整数倍
        x = self.pad(x)

        # 3D卷积进行patch embedding
        x = self.proj(x)

        # 可选的归一化
        if self.norm:
            x = rearrange(x, 'b c l h w -> b l h w c')
            x = self.norm(x)
            x = rearrange(x, 'b l h w c -> b c l h w')
        return x


# ==================== Patch Recovery模块（使用einops改进） ====================

class PatchRecovery2D(nn.Module):
    """
    将Patch Embedding恢复为2D图像
    使用转置卷积上采样
    """
    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        """
        输入: (B, in_chans, H_p, W_p)
        输出: (B, out_chans, H, W)
        """
        output = self.conv(x)
        _, _, H, W = output.shape

        # 中心裁剪到目标尺寸
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)
        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[
            :, :, padding_top : H - padding_bottom, padding_left : W - padding_right
        ]


class PatchRecovery3D(nn.Module):
    """
    将Patch Embedding恢复为3D图像
    使用3D转置卷积上采样
    """
    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        """
        输入: (B, in_chans, L_p, H_p, W_p)
        输出: (B, out_chans, L, H, W)
        """
        output = self.conv(x)
        _, _, Pl, Lat, Lon = output.shape

        # 中心裁剪到目标尺寸
        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[
            :, :,
            padding_front : Pl - padding_back,
            padding_top : Lat - padding_bottom,
            padding_left : Lon - padding_right,
        ]


# ==================== 上下采样模块（使用einops改进） ====================

class UpSample3D(nn.Module):
    """
    3D上采样操作
    使用PixelShuffle的思想，通过线性层扩展通道后重排实现上采样
    """
    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        输入: (B, N, C) 其中 N = in_pl * in_lat * in_lon
        输出: (B, N_out, C_out) 其中 N_out = out_pl * out_lat * out_lon
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        # 线性层扩展通道: (B, N, C) -> (B, N, C*4)
        x = self.linear1(x)

        # 使用einops进行PixelShuffle式重排
        # (B, N, C*4) -> (B, in_pl, in_lat, in_lon, 2, 2, C//2)
        x = rearrange(x, 'b (pl lat lon) (h w c) -> b pl (lat h) (lon w) c',
                     pl=in_pl, lat=in_lat, lon=in_lon, h=2, w=2)

        # 计算padding进行中心裁剪
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 裁剪到目标尺寸
        x = x[:, :out_pl,
              pad_top : 2 * in_lat - pad_bottom,
              pad_left : 2 * in_lon - pad_right, :]

        # 展平并进行最终变换
        x = rearrange(x, 'b pl lat lon c -> b (pl lat lon) c')
        x = self.norm(x)
        x = self.linear2(x)
        return x


class UpSample2D(nn.Module):
    """
    2D上采样操作
    """
    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        输入: (B, N, C) 其中 N = in_lat * in_lon
        输出: (B, N_out, C_out)
        """
        B, N, C = x.shape
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution

        # 线性层扩展通道
        x = self.linear1(x)

        # 使用einops进行PixelShuffle式重排
        x = rearrange(x, 'b (lat lon) (h w c) -> b (lat h) (lon w) c',
                     lat=in_lat, lon=in_lon, h=2, w=2)

        # 中心裁剪
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[:, pad_top : 2 * in_lat - pad_bottom,
              pad_left : 2 * in_lon - pad_right, :]

        # 展平并进行最终变换
        x = rearrange(x, 'b lat lon c -> b (lat lon) c')
        x = self.norm(x)
        x = self.linear2(x)
        return x


class DownSample3D(nn.Module):
    """
    3D下采样操作
    将2x2空间块合并到通道维度实现下采样
    """
    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        # 计算padding
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left
        pad_front = pad_back = 0

        self.pad = nn.ConstantPad3d(
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back), value=0
        )

    def forward(self, x):
        """
        输入: (B, N, C) 其中 N = in_pl * in_lat * in_lon
        输出: (B, N_out, 2C)
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        # 重排为3D网格
        x = rearrange(x, 'b (pl lat lon) c -> b pl lat lon c',
                     pl=in_pl, lat=in_lat, lon=in_lon)

        # Padding
        x = rearrange(x, 'b pl lat lon c -> b c pl lat lon')
        x = self.pad(x)
        x = rearrange(x, 'b c pl lat lon -> b pl lat lon c')

        # 下采样：将2x2块合并到通道
        x = rearrange(x, 'b pl (olat h) (olon w) c -> b pl olat olon (h w c)',
                     olat=out_lat, olon=out_lon, h=2, w=2)

        # 展平并降维
        x = rearrange(x, 'b pl olat olon c -> b (pl olat olon) c')
        x = self.norm(x)
        x = self.linear(x)
        return x


class DownSample2D(nn.Module):
    """
    2D下采样操作
    """
    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution

        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        self.pad = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), value=0)

    def forward(self, x: torch.Tensor):
        """
        输入: (B, N, C) 其中 N = in_lat * in_lon
        输出: (B, N_out, 2C)
        """
        B, N, C = x.shape
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution

        # 重排为2D网格
        x = rearrange(x, 'b (lat lon) c -> b lat lon c',
                     lat=in_lat, lon=in_lon)

        # Padding
        x = rearrange(x, 'b lat lon c -> b c lat lon')
        x = self.pad(x)
        x = rearrange(x, 'b c lat lon -> b lat lon c')

        # 下采样：将2x2块合并到通道
        x = rearrange(x, 'b (olat h) (olon w) c -> b olat olon (h w c)',
                     olat=out_lat, olon=out_lon, h=2, w=2)

        # 展平并降维
        x = rearrange(x, 'b olat olon c -> b (olat olon) c')
        x = self.norm(x)
        x = self.linear(x)
        return x


# ==================== 位置编码和窗口操作函数（从Pangu复用） ====================

def get_earth_position_index(window_size, ndim=3):
    """
    构建地球位置索引，用于位置偏置的重用
    考虑地球的球形几何特性
    """
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
    elif ndim == 2:
        win_lat, win_lon = window_size

    if ndim == 3:
        # 气压层维度的索引
        coords_zi = torch.arange(win_pl)
        coords_zj = -torch.arange(win_pl) * win_pl

    # 纬度维度的索引
    coords_hi = torch.arange(win_lat)
    coords_hj = -torch.arange(win_lat) * win_lat

    # 经度维度的索引（考虑循环边界）
    coords_w = torch.arange(win_lon)

    # 创建网格并计算相对位置
    if ndim == 3:
        coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w], indexing='ij'))
        coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w], indexing='ij'))
    elif ndim == 2:
        coords_1 = torch.stack(torch.meshgrid([coords_hi, coords_w], indexing='ij'))
        coords_2 = torch.stack(torch.meshgrid([coords_hj, coords_w], indexing='ij'))

    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    # 偏移索引使其从0开始
    if ndim == 3:
        coords[:, :, 2] += win_lon - 1
        coords[:, :, 1] *= 2 * win_lon - 1
        coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat
    elif ndim == 2:
        coords[:, :, 1] += win_lon - 1
        coords[:, :, 0] *= 2 * win_lon - 1

    position_index = coords.sum(-1)
    return position_index


def get_pad3d(input_resolution, window_size):
    """计算3D padding"""
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0

    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return (padding_left, padding_right, padding_top, padding_bottom,
            padding_front, padding_back)


def get_pad2d(input_resolution, window_size):
    """计算2D padding"""
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[:4]


def crop3d(x: torch.Tensor, resolution):
    """3D中心裁剪"""
    _, _, Pl, Lat, Lon = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front
    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top
    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left

    return x[:, :,
            padding_front : Pl - padding_back,
            padding_top : Lat - padding_bottom,
            padding_left : Lon - padding_right]


def crop2d(x: torch.Tensor, resolution):
    """2D中心裁剪"""
    _, _, Lat, Lon = x.shape
    lat_pad = Lat - resolution[0]
    lon_pad = Lon - resolution[1]

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top
    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left

    return x[:, :,
            padding_top : Lat - padding_bottom,
            padding_left : Lon - padding_right]


def window_partition(x: torch.Tensor, window_size, ndim=3):
    """
    将特征图划分为不重叠的窗口
    支持2D和3D
    """
    if ndim == 3:
        B, Pl, Lat, Lon, C = x.shape
        win_pl, win_lat, win_lon = window_size

        # 使用einops重排
        windows = rearrange(x,
            'b (npl wpl) (nlat wlat) (nlon wlon) c -> (b nlon) (npl nlat) wpl wlat wlon c',
            wpl=win_pl, wlat=win_lat, wlon=win_lon)
        return windows
    elif ndim == 2:
        B, Lat, Lon, C = x.shape
        win_lat, win_lon = window_size

        windows = rearrange(x,
            'b (nlat wlat) (nlon wlon) c -> (b nlon) nlat wlat wlon c',
            wlat=win_lat, wlon=win_lon)
        return windows


def window_reverse(windows, window_size, Pl=1, Lat=1, Lon=1, ndim=3):
    """
    将窗口合并回完整的特征图
    """
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))

        x = rearrange(windows,
            '(b nlon) (npl nlat) wpl wlat wlon c -> b (npl wpl) (nlat wlat) (nlon wlon) c',
            b=B, nlon=Lon//win_lon, npl=Pl//win_pl, nlat=Lat//win_lat)
        return x
    elif ndim == 2:
        win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))

        x = rearrange(windows,
            '(b nlon) nlat wlat wlon c -> b (nlat wlat) (nlon wlon) c',
            b=B, nlon=Lon//win_lon, nlat=Lat//win_lat)
        return x


def get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3):
    """
    生成shifted window attention的mask
    用于防止不同窗口之间的信息交互
    """
    if ndim == 3:
        Pl, Lat, Lon = input_resolution
        win_pl, win_lat, win_lon = window_size
        shift_pl, shift_lat, shift_lon = shift_size
        img_mask = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))
    elif ndim == 2:
        Lat, Lon = input_resolution
        win_lat, win_lon = window_size
        shift_lat, shift_lon = shift_size
        img_mask = torch.zeros((1, Lat, Lon + shift_lon, 1))

    # 创建区域索引
    if ndim == 3:
        pl_slices = (slice(0, -win_pl), slice(-win_pl, -shift_pl), slice(-shift_pl, None))
    lat_slices = (slice(0, -win_lat), slice(-win_lat, -shift_lat), slice(-shift_lat, None))
    lon_slices = (slice(0, -win_lon), slice(-win_lon, -shift_lon), slice(-shift_lon, None))

    cnt = 0
    if ndim == 3:
        for pl in pl_slices:
            for lat in lat_slices:
                for lon in lon_slices:
                    img_mask[:, pl, lat, lon, :] = cnt
                    cnt += 1
        img_mask = img_mask[:, :, :, :Lon, :]
    elif ndim == 2:
        for lat in lat_slices:
            for lon in lon_slices:
                img_mask[:, lat, lon, :] = cnt
                cnt += 1
        img_mask = img_mask[:, :, :Lon, :]

    # 生成attention mask
    mask_windows = window_partition(img_mask, window_size, ndim=ndim)

    if ndim == 3:
        win_total = win_pl * win_lat * win_lon
    elif ndim == 2:
        win_total = win_lat * win_lon

    mask_windows = mask_windows.reshape(
        mask_windows.shape[0], mask_windows.shape[1], win_total
    )
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


# ==================== Earth Attention模块（使用einops改进） ====================

class EarthAttention2D(nn.Module):
    """
    2D窗口注意力机制，带地球位置偏置
    考虑地球的球形几何特性，经度方向是循环的
    """
    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (win_lat, win_lon)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # 窗口类型数量（纬度方向的窗口数）
        self.type_of_windows = input_resolution[0] // window_size[0]

        # 地球位置偏置表
        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2) * (window_size[1] * 2 - 1),
                self.type_of_windows,
                num_heads,
            )
        )

        # 注册位置索引
        earth_position_index = get_earth_position_index(window_size, ndim=2)
        self.register_buffer("earth_position_index", earth_position_index)

        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化位置偏置
        self.earth_position_bias_table = trunc_normal_(
            self.earth_position_bias_table, std=0.02
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        """
        输入: (B_, nW_, N, C)
        其中: B_=B*num_lon, nW_=num_lat, N=win_lat*win_lon
        """
        B_, nW_, N, C = x.shape

        # 生成Q, K, V并重排为多头格式
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b nw n (three h d) -> three b h nw n d',
                       three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # 添加地球位置偏置
        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            self.type_of_windows,
            -1,
        )
        earth_position_bias = rearrange(earth_position_bias, 'n1 n2 nw h -> h nw n1 n2')
        attn = attn + earth_position_bias.unsqueeze(0)

        # 应用mask（如果有）
        if mask is not None:
            nLon = mask.shape[0]
            attn = rearrange(attn, '(b nlon) h nw n1 n2 -> b nlon h nw n1 n2', nlon=nLon)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = rearrange(attn, 'b nlon h nw n1 n2 -> (b nlon) h nw n1 n2')
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # 加权求和并输出投影
        x = attn @ v
        x = rearrange(x, 'b h nw n d -> b nw n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EarthAttention3D(nn.Module):
    """
    3D窗口注意力机制，带地球位置偏置
    处理包含气压层维度的3D气象数据
    """
    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (win_pl, win_lat, win_lon)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # 窗口类型数量
        self.type_of_windows = (input_resolution[0] // window_size[0]) * (
            input_resolution[1] // window_size[1]
        )

        # 地球位置偏置表
        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1),
                self.type_of_windows,
                num_heads,
            )
        )

        # 注册位置索引
        earth_position_index = get_earth_position_index(window_size)
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.earth_position_bias_table = trunc_normal_(
            self.earth_position_bias_table, std=0.02
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        """
        输入: (B_, nW_, N, C)
        其中: B_=B*num_lon, nW_=num_pl*num_lat, N=win_pl*win_lat*win_lon
        """
        B_, nW_, N, C = x.shape

        # 生成Q, K, V
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b nw n (three h d) -> three b h nw n d',
                       three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # 添加地球位置偏置
        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows,
            -1,
        )
        earth_position_bias = rearrange(earth_position_bias, 'n1 n2 nw h -> h nw n1 n2')
        attn = attn + earth_position_bias.unsqueeze(0)

        # 应用mask
        if mask is not None:
            nLon = mask.shape[0]
            attn = rearrange(attn, '(b nlon) h nw n1 n2 -> b nlon h nw n1 n2', nlon=nLon)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = rearrange(attn, 'b nlon h nw n1 n2 -> (b nlon) h nw n1 n2')
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # 输出
        x = attn @ v
        x = rearrange(x, 'b h nw n d -> b nw n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ==================== Transformer Block模块 ====================

class Transformer3DBlock(nn.Module):
    """
    3D Transformer块
    包含窗口注意力和FFN，支持shifted window
    """
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=None,
        shift_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 第一个LayerNorm
        self.norm1 = norm_layer(dim)

        # Padding
        padding = get_pad3d(input_resolution, window_size)
        self.pad = nn.ConstantPad3d(padding, value=0)

        # 计算padding后的分辨率
        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[-1] + padding[-2]
        pad_resolution[1] += padding[2] + padding[3]
        pad_resolution[2] += padding[0] + padding[1]

        # 注意力层
        self.attn = EarthAttention3D(
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 第二个LayerNorm和FFN
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # 判断是否需要shifted window
        shift_pl, shift_lat, shift_lon = self.shift_size
        self.roll = shift_pl and shift_lon and shift_lat

        # 生成attention mask
        if self.roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        """
        输入: (B, L, C) 其中 L = Pl * Lat * Lon
        """
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape

        # 保存残差连接
        shortcut = x

        # LayerNorm
        x = self.norm1(x)

        # 重排为3D
        x = rearrange(x, 'b (pl lat lon) c -> b pl lat lon c',
                     pl=Pl, lat=Lat, lon=Lon)

        # Padding
        x = rearrange(x, 'b pl lat lon c -> b c pl lat lon')
        x = self.pad(x)
        _, _, Pl_pad, Lat_pad, Lon_pad = x.shape
        x = rearrange(x, 'b c pl lat lon -> b pl lat lon c')

        # Shifted Window处理
        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            # 循环移位
            shifted_x = torch.roll(
                x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3)
            )
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        # 展平窗口内部
        win_pl, win_lat, win_lon = self.window_size
        x_windows = rearrange(x_windows,
                             'b nw wpl wlat wlon c -> b nw (wpl wlat wlon) c')

        # 窗口注意力
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 恢复窗口形状
        attn_windows = rearrange(attn_windows,
                                'b nw (wpl wlat wlon) c -> b nw wpl wlat wlon c',
                                wpl=win_pl, wlat=win_lat, wlon=win_lon)

        # 窗口反向合并
        if self.roll:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad
            )
            # 反向循环移位
            x = torch.roll(
                shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3)
            )
        else:
            x = window_reverse(
                attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad
            )

        # Crop去除padding
        x = rearrange(x, 'b pl lat lon c -> b c pl lat lon')
        x = crop3d(x, self.input_resolution)
        x = rearrange(x, 'b c pl lat lon -> b pl lat lon c')

        # 展平回序列格式
        x = rearrange(x, 'b pl lat lon c -> b (pl lat lon) c')

        # 第一个残差连接
        x = shortcut + self.drop_path(x)

        # FFN和第二个残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Transformer2DBlock(nn.Module):
    """
    2D Transformer块
    """
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=None,
        shift_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        window_size = (6, 12) if window_size is None else window_size
        shift_size = (3, 6) if shift_size is None else shift_size

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        padding = get_pad2d(input_resolution, window_size)
        self.pad = nn.ConstantPad2d(padding, value=0)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[2] + padding[3]
        pad_resolution[1] += padding[0] + padding[1]

        self.attn = EarthAttention2D(
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        shift_lat, shift_lon = self.shift_size
        self.roll = shift_lon and shift_lat

        if self.roll:
            attn_mask = get_shift_window_mask(
                pad_resolution, window_size, shift_size, ndim=2
            )
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        """
        输入: (B, L, C) 其中 L = Lat * Lon
        """
        Lat, Lon = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)

        # 重排为2D
        x = rearrange(x, 'b (lat lon) c -> b lat lon c', lat=Lat, lon=Lon)

        # Padding
        x = rearrange(x, 'b lat lon c -> b c lat lon')
        x = self.pad(x)
        _, _, Lat_pad, Lon_pad = x.shape
        x = rearrange(x, 'b c lat lon -> b lat lon c')

        # Shifted Window
        shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_lat, -shift_lon), dims=(1, 2))
            x_windows = window_partition(shifted_x, self.window_size, ndim=2)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size, ndim=2)

        # 窗口注意力
        win_lat, win_lon = self.window_size
        x_windows = rearrange(x_windows, 'b nw wlat wlon c -> b nw (wlat wlon) c')
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = rearrange(attn_windows,
                                'b nw (wlat wlon) c -> b nw wlat wlon c',
                                wlat=win_lat, wlon=win_lon)

        # 窗口反向合并
        if self.roll:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Lat=Lat_pad, Lon=Lon_pad, ndim=2
            )
            x = torch.roll(shifted_x, shifts=(shift_lat, shift_lon), dims=(1, 2))
        else:
            x = window_reverse(
                attn_windows, self.window_size, Lat=Lat_pad, Lon=Lon_pad, ndim=2
            )

        # Crop
        x = rearrange(x, 'b lat lon c -> b c lat lon')
        x = crop2d(x, self.input_resolution)
        x = rearrange(x, 'b c lat lon -> b lat lon c')

        # 展平
        x = rearrange(x, 'b lat lon c -> b (lat lon) c')

        # 残差连接
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ==================== Layer模块 ====================

class FuserLayer(nn.Module):
    """
    融合层：包含多个3D Transformer块
    用于融合不同变量的特征
    """
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # 构建多个Transformer块
        self.blocks = nn.ModuleList([
            Transformer3DBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else None,  # 交替使用shifted window
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        """依次通过所有Transformer块"""
        for blk in self.blocks:
            x = blk(x)
        return x


class EncoderLayer(nn.Module):
    """
    编码器层：处理2D输入
    包含patch embedding、Transformer块、下采样
    """
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        dim,
        input_resolution,
        middle_resolution,
        depth,
        depth_middle,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.depth_middle = depth_middle

        # 处理drop_path参数
        if isinstance(drop_path, Sequence):
            drop_path_middle = drop_path[depth:]
            drop_path = drop_path[:depth]
        else:
            drop_path_middle = drop_path

        # 处理num_heads参数
        if isinstance(num_heads, Sequence):
            num_heads_middle = num_heads[1]
            num_heads = num_heads[0]
        else:
            num_heads_middle = num_heads

        # Patch Embedding
        self.patchembed2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=dim,
        )

        # 第一组Transformer块
        self.blocks = nn.ModuleList([
            Transformer2DBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if i % 2 == 0 else None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # 下采样
        self.downsample = DownSample2D(
            in_dim=dim,
            input_resolution=input_resolution,
            output_resolution=middle_resolution,
        )

        # 第二组Transformer块（更深）
        self.blocks_middle = nn.ModuleList([
            Transformer2DBlock(
                dim=dim * 2,
                input_resolution=middle_resolution,
                num_heads=num_heads_middle,
                window_size=window_size,
                shift_size=(0, 0) if i % 2 == 0 else None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_middle[i] if isinstance(drop_path_middle, Sequence) else drop_path_middle,
                norm_layer=norm_layer,
            )
            for i in range(depth_middle)
        ])

    def forward(self, x):
        """
        输入: (B, C, H, W)
        输出: (特征, skip连接)
        """
        # Patch Embedding
        x = self.patchembed2d(x)
        B, C, Lat, Lon = x.shape
        x = rearrange(x, 'b c lat lon -> b (lat lon) c')

        # 第一组Transformer块
        for blk in self.blocks:
            x = blk(x)

        # 保存skip连接
        skip = rearrange(x, 'b (lat lon) c -> b lat lon c', lat=Lat, lon=Lon)

        # 下采样
        x = self.downsample(x)

        # 第二组Transformer块
        for blk in self.blocks_middle:
            x = blk(x)

        return x, skip


class DecoderLayer(nn.Module):
    """
    解码器层：处理2D输出
    包含Transformer块、上采样、patch recovery
    """
    def __init__(
        self,
        img_size,
        patch_size,
        out_chans,
        dim,
        output_resolution,
        middle_resolution,
        depth,
        depth_middle,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.out_chans = out_chans
        self.dim = dim
        self.output_resolution = output_resolution
        self.depth = depth
        self.depth_middle = depth_middle

        # 处理参数
        if isinstance(drop_path, Sequence):
            drop_path_middle = drop_path[depth:]
            drop_path = drop_path[:depth]
        else:
            drop_path_middle = drop_path

        if isinstance(num_heads, Sequence):
            num_heads_middle = num_heads[1]
            num_heads = num_heads[0]
        else:
            num_heads_middle = num_heads

        # 第一组Transformer块
        self.blocks_middle = nn.ModuleList([
            Transformer2DBlock(
                dim=dim * 2,
                input_resolution=middle_resolution,
                num_heads=num_heads_middle,
                window_size=window_size,
                shift_size=(0, 0) if i % 2 == 0 else None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_middle[i] if isinstance(drop_path_middle, Sequence) else drop_path_middle,
                norm_layer=norm_layer,
            )
            for i in range(depth_middle)
        ])

        # 上采样
        self.upsample = UpSample2D(
            in_dim=dim * 2,
            out_dim=dim,
            input_resolution=middle_resolution,
            output_resolution=output_resolution,
        )

        # 第二组Transformer块
        self.blocks = nn.ModuleList([
            Transformer2DBlock(
                dim=dim,
                input_resolution=output_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if i % 2 == 0 else None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # Patch Recovery
        self.patchrecovery2d = PatchRecovery2D(img_size, patch_size, 2 * dim, out_chans)

    def forward(self, x, skip):
        """
        输入: x (特征), skip (skip连接)
        输出: (B, out_chans, H, W)
        """
        B, Lat, Lon, C = skip.shape

        # 第一组Transformer块
        for blk in self.blocks_middle:
            x = blk(x)

        # 上采样
        x = self.upsample(x)

        # 第二组Transformer块
        for blk in self.blocks:
            x = blk(x)

        # 连接skip connection
        skip_flat = rearrange(skip, 'b lat lon c -> b (lat lon) c')
        output = torch.cat([x, skip_flat], dim=-1)

        # Patch Recovery
        output = rearrange(output, 'b (lat lon) c -> b c lat lon', lat=Lat, lon=Lon)
        output = self.patchrecovery2d(output)

        return output


# ==================== 主模型Fengwu ====================

class Fengwu(nn.Module):
    """
    FengWu气象预测模型

    FengWu是一个基于Transformer的天气预报模型，能够进行10天以上的中期天气预报。
    模型采用编码器-融合器-解码器架构：
    - 编码器：分别处理地表和不同高度层的气象变量
    - 融合器：在3D空间中融合所有变量的特征
    - 解码器：生成下一时刻的预报

    输入格式: (B, 69, H, W)
    - 前65通道：5个高空变量 × 13个气压层
    - 后4通道：4个地表变量

    输出格式: (B, 69, H, W) - 预测的下一时刻气象场
    """
    def __init__(
        self,
        in_shape=(1, 69, 120, 240),
        pressure_level=13,
        embed_dim=192,
        patch_size=(4, 4),
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
        **kwargs
    ):
        super().__init__()

        # 获取图像尺寸
        img_size = (in_shape[2], in_shape[3])

        # Drop path率
        drop_path = np.linspace(0, 0.2, 8).tolist()
        drop_path_fuser = [0.2] * 6

        # 计算不同分辨率
        resolution_down1 = (
            math.ceil(img_size[0] / patch_size[0]),
            math.ceil(img_size[1] / patch_size[1]),
        )
        resolution_down2 = (
            math.ceil(resolution_down1[0] / 2),
            math.ceil(resolution_down1[1] / 2),
        )
        resolution = (resolution_down1, resolution_down2)

        # ==================== 编码器 ====================
        # 地表变量编码器
        self.encoder_surface = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=4,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        # 高空变量编码器（z: 位势高度）
        self.encoder_z = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        # 高空变量编码器（r: 相对湿度）
        self.encoder_r = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        # 高空变量编码器（u: 纬向风）
        self.encoder_u = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        # 高空变量编码器（v: 经向风）
        self.encoder_v = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        # 高空变量编码器（t: 温度）
        self.encoder_t = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        # ==================== 融合器 ====================
        self.fuser = FuserLayer(
            dim=embed_dim * 2,
            input_resolution=(6, resolution[1][0], resolution[1][1]),  # 6个变量
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path_fuser,
        )

        # ==================== 解码器 ====================
        # 地表变量解码器
        self.decoder_surface = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=4,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        # 高空变量解码器
        self.decoder_z = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        self.decoder_r = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        self.decoder_u = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        self.decoder_v = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        self.decoder_t = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

    def forward(self, x):
        """
        前向传播

        输入: (B, 69, H, W)
        - 通道0-12: z (位势高度) 13个气压层
        - 通道13-25: r (相对湿度) 13个气压层
        - 通道26-38: t (温度) 13个气压层
        - 通道39-51: u (纬向风) 13个气压层
        - 通道52-64: v (经向风) 13个气压层
        - 通道65-68: surface (地表变量) 4个通道

        输出: (B, 69, H, W) - 相同格式的预测结果
        """
        # ==================== 数据分离 ====================
        # 提取地表变量
        surface = x[:, 65:69, :, :]

        # 提取高空变量（按照输入格式）
        z = x[:, 0:13, :, :]    # 位势高度
        r = x[:, 13:26, :, :]   # 相对湿度
        t = x[:, 26:39, :, :]   # 温度
        u = x[:, 39:52, :, :]   # 纬向风
        v = x[:, 52:65, :, :]   # 经向风

        # ==================== 编码 ====================
        # 对每个变量分别编码
        surface_enc, skip_surface = self.encoder_surface(surface)
        z_enc, skip_z = self.encoder_z(z)
        r_enc, skip_r = self.encoder_r(r)
        u_enc, skip_u = self.encoder_u(u)
        v_enc, skip_v = self.encoder_v(v)
        t_enc, skip_t = self.encoder_t(t)

        # ==================== 融合 ====================
        # 将所有编码后的特征堆叠为3D张量
        # 每个编码器输出: (B, N, C) 其中N = lat_p * lon_p
        # 堆叠后: (B, 6, N, C)
        x_fused = torch.stack([
            surface_enc,
            z_enc,
            r_enc,
            u_enc,
            v_enc,
            t_enc
        ], dim=1)

        B, num_vars, L_SIZE, C = x_fused.shape

        # 重排为序列格式进行3D attention
        x_fused = rearrange(x_fused, 'b nv l c -> b (nv l) c')

        # 通过融合器
        x_fused = self.fuser(x_fused)

        # 重排回分离的变量
        x_fused = rearrange(x_fused, 'b (nv l) c -> b nv l c', nv=num_vars)

        # 分离融合后的特征
        surface_fused = x_fused[:, 0, :, :]
        z_fused = x_fused[:, 1, :, :]
        r_fused = x_fused[:, 2, :, :]
        u_fused = x_fused[:, 3, :, :]
        v_fused = x_fused[:, 4, :, :]
        t_fused = x_fused[:, 5, :, :]

        # ==================== 解码 ====================
        # 对每个变量分别解码
        surface_out = self.decoder_surface(surface_fused, skip_surface)
        z_out = self.decoder_z(z_fused, skip_z)
        r_out = self.decoder_r(r_fused, skip_r)
        u_out = self.decoder_u(u_fused, skip_u)
        v_out = self.decoder_v(v_fused, skip_v)
        t_out = self.decoder_t(t_fused, skip_t)

        # ==================== 输出拼接 ====================
        # 按照输入格式拼接输出
        final_output = torch.cat([
            z_out,      # 通道0-12
            r_out,      # 通道13-25
            t_out,      # 通道26-38
            u_out,      # 通道39-51
            v_out,      # 通道52-64
            surface_out # 通道65-68
        ], dim=1)

        return final_output


if __name__ == '__main__':
    """测试代码"""
    # 创建模型
    model = Fengwu()

    # 创建测试输入
    inputs = torch.randn(1, 69, 120, 240)

    # 前向传播
    output = model(inputs)

    # 打印输入输出形状
    print(f"输入形状: {inputs.shape}")
    print(f"输出形状: {output.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
