"""
Pangu1 Model - 仅使用2D Transformer块的版本
适配海洋流速预测任务：输入多帧2D数据，输出多帧2D数据
"""

import torch
import torch.nn as nn
from einops import rearrange
import math
import numpy as np
from .base import BaseModel


# ==================== 辅助函数 ====================
def get_pad2d(input_resolution, window_size):
    """计算2D padding"""
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)

    padding_left = padding_right = padding_top = padding_bottom = 0

    h_remainder = input_resolution[1] % window_size[1]
    w_remainder = input_resolution[2] % window_size[2]

    if h_remainder:
        h_pad = window_size[1] - h_remainder
        padding_top = h_pad // 2
        padding_bottom = h_pad - padding_top

    if w_remainder:
        w_pad = window_size[2] - w_remainder
        padding_left = w_pad // 2
        padding_right = w_pad - padding_left

    return (padding_left, padding_right, padding_top, padding_bottom)


def crop2d(x, resolution):
    """裁剪2D tensor到指定分辨率"""
    _, _, H, W = x.shape
    h_pad = H - resolution[0]
    w_pad = W - resolution[1]

    padding_top = h_pad // 2
    padding_bottom = h_pad - padding_top
    padding_left = w_pad // 2
    padding_right = w_pad - padding_left

    return x[:, :, padding_top:H-padding_bottom, padding_left:W-padding_right]


def window_partition(x, window_size):
    """将输入分割为窗口"""
    B, H, W, C = x.shape
    win_h, win_w = window_size

    x = x.view(B, H // win_h, win_h, W // win_w, win_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """将窗口恢复为原始形状"""
    win_h, win_w = window_size
    B = int(windows.shape[0] / (H * W / win_h / win_w))

    x = windows.view(B, H // win_h, W // win_w, win_h, win_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_earth_position_index(window_size):
    """生成Earth Position Index（2D版本）"""
    win_h, win_w = window_size

    # 纬度和经度索引
    coords_h = torch.arange(win_h)
    coords_w = torch.arange(win_w)
    coords_hh = -torch.arange(win_h) * win_h

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords_flatten = torch.flatten(coords, 1)

    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()

    # 偏移以确保索引从0开始
    relative_coords[:, :, 0] *= 2 * win_w - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_position_index = relative_coords.sum(-1)

    return relative_position_index


# ==================== 模块定义 ====================
class PatchEmbed2D(nn.Module):
    """2D Patch Embedding"""
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # Padding
        height, width = img_size
        h_patch, w_patch = patch_size

        padding_left = padding_right = padding_top = padding_bottom = 0

        h_remainder = height % h_patch
        w_remainder = width % w_patch

        if h_remainder:
            h_pad = h_patch - h_remainder
            padding_top = h_pad // 2
            padding_bottom = h_pad - padding_top

        if w_remainder:
            w_pad = w_patch - w_remainder
            padding_left = w_pad // 2
            padding_right = w_pad - padding_left

        self.pad = nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.pad(x)
        x = self.proj(x)
        # (B, C, H, W) -> (B, H, W, C)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return x


class PatchRecovery2D(nn.Module):
    """2D Patch Recovery"""
    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        output = self.conv(x)
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = h_pad - padding_top
        padding_left = w_pad // 2
        padding_right = w_pad - padding_left

        return output[:, :, padding_top:H-padding_bottom, padding_left:W-padding_right]


class EarthAttention2D(nn.Module):
    """Earth-aware 2D Attention"""
    def __init__(self, dim, input_resolution, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.type_of_windows = input_resolution[0] // window_size[0]

        # Earth position bias
        self.earth_position_bias_table = nn.Parameter(
            torch.zeros((window_size[0]**2) * (window_size[1]*2-1),
                       self.type_of_windows, num_heads)
        )
        nn.init.trunc_normal_(self.earth_position_bias_table, std=0.02)

        earth_position_index = get_earth_position_index(window_size)
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Earth position bias
        # B_ = batch_size * num_h_windows * num_w_windows
        # 需要为每个窗口选择对应的位置偏置

        # 获取位置偏置表
        earth_position_bias = self.earth_position_bias_table[self.earth_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            self.type_of_windows, -1
        )

        # 计算实际的窗口数量
        batch_size = B_ // (N)  # B_ / (window_size[0] * window_size[1])

        # 为每个窗口选择对应纬度的偏置
        # 假设所有窗口使用相同的偏置（中间纬度）
        middle_latitude = self.type_of_windows // 2
        earth_position_bias = earth_position_bias[:, :, middle_latitude, :]  # (N, N, num_heads)
        earth_position_bias = earth_position_bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, N, N)

        attn = attn + earth_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock2D(nn.Module):
    """2D Transformer Block"""
    def __init__(self, dim, input_resolution, num_heads, window_size,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = EarthAttention2D(
            dim, input_resolution, window_size, num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Padding
        self.padding = get_pad2d(input_resolution, window_size)
        self.pad = nn.ZeroPad2d(self.padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.pad(x)
        _, _, Hp, Wp = x.shape
        x = rearrange(x, 'b c h w -> b h w c')

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Crop
        x = rearrange(x, 'b h w c -> b c h w')
        x = crop2d(x, self.input_resolution)
        x = rearrange(x, 'b c h w -> b h w c')
        x = x.reshape(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP层"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DownSample2D(nn.Module):
    """2D下采样模块"""
    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_h, in_w = input_resolution
        out_h, out_w = output_resolution

        h_pad = out_h * 2 - in_h
        w_pad = out_w * 2 - in_w

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        self.pad = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        out_h, out_w = self.output_resolution

        x = x.view(B, H, W, C)

        # Pad and downsample
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.pad(x)
        x = rearrange(x, 'b c h w -> b h w c')

        x = rearrange(x, 'b (h p1) (w p2) c -> b h w (p1 p2 c)', p1=2, p2=2, h=out_h, w=out_w)
        x = x.view(B, out_h * out_w, 4 * C)

        x = self.norm(x)
        x = self.linear(x)

        return x


class UpSample2D(nn.Module):
    """2D上采样模块"""
    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        out_h, out_w = self.output_resolution

        x = self.linear1(x)
        x = x.view(B, H, W, -1)

        # PixelShuffle-like upsampling
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2)

        # Crop to target resolution
        pad_h = H * 2 - out_h
        pad_w = W * 2 - out_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[:, pad_top:2*H-pad_bottom, pad_left:2*W-pad_right, :]
        x = x.reshape(B, out_h * out_w, -1)

        x = self.norm(x)
        x = self.linear2(x)

        return x


# ==================== 主模型 ====================
class OceanPangu1(BaseModel):
    """
    Pangu1 - 仅使用2D Transformer块的版本
    适配海洋流速预测任务
    """
    def __init__(self, args):
        super(BaseModel, self).__init__()

        # 任务参数
        self.input_len = args.get('input_len', 7)
        self.output_len = args.get('output_len', 1)
        self.in_channels = args.get('in_channels', 2)

        # 模型参数
        img_size = args.get('img_size', (240, 240))
        patch_size = args.get('patch_size', (4, 4))
        embed_dim = args.get('embed_dim', 192)
        depths = args.get('depths', [2, 6, 6, 2])
        num_heads = args.get('num_heads', [6, 12, 12, 6])
        window_size = args.get('window_size', (8, 8))
        mlp_ratio = args.get('mlp_ratio', 4.)
        drop_rate = args.get('drop_rate', 0.)
        attn_drop_rate = args.get('attn_drop_rate', 0.)
        drop_path_rate = args.get('drop_path_rate', 0.2)

        # 时序融合：将多帧合并为通道
        self.temporal_embed = nn.Conv2d(
            self.input_len * self.in_channels,
            embed_dim // 2,
            kernel_size=1
        )

        # Patch Embedding
        self.patch_embed = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim // 2,
            embed_dim=embed_dim
        )

        # 计算patch后的分辨率
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()

        # Layer 1 (高分辨率)
        layer1 = nn.ModuleList([
            TransformerBlock2D(
                dim=embed_dim,
                input_resolution=patches_resolution,
                num_heads=num_heads[0],
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size[0] // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i]
            ) for i in range(depths[0])
        ])
        self.layers.append(layer1)

        # Downsample
        downsample_resolution = [patches_resolution[0] // 2, patches_resolution[1] // 2]
        self.downsample = DownSample2D(
            embed_dim,
            patches_resolution,
            downsample_resolution
        )

        # Layer 2 & 3 (低分辨率)
        cur_depth = depths[0]
        for layer_idx in range(1, 3):
            layer = nn.ModuleList([
                TransformerBlock2D(
                    dim=embed_dim * 2,
                    input_resolution=downsample_resolution,
                    num_heads=num_heads[layer_idx],
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size[0] // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur_depth + i]
                ) for i in range(depths[layer_idx])
            ])
            self.layers.append(layer)
            cur_depth += depths[layer_idx]

        # Upsample
        self.upsample = UpSample2D(
            embed_dim * 2,
            embed_dim,
            downsample_resolution,
            patches_resolution
        )

        # Layer 4 (高分辨率)
        layer4 = nn.ModuleList([
            TransformerBlock2D(
                dim=embed_dim,
                input_resolution=patches_resolution,
                num_heads=num_heads[3],
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size[0] // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur_depth + i]
            ) for i in range(depths[3])
        ])
        self.layers.append(layer4)

        # Patch Recovery
        self.patch_recovery = PatchRecovery2D(
            img_size,
            patch_size,
            embed_dim * 2,  # skip connection
            self.output_len * self.in_channels
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        输入: (B, T_in, C, H, W)
        输出: (B, T_out, C, H, W)
        """
        B, T_in, C, H, W = x.shape

        # 时序融合: (B, T_in, C, H, W) -> (B, T_in*C, H, W)
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.temporal_embed(x)

        # Patch Embedding: (B, embed_dim//2, H, W) -> (B, H', W', embed_dim)
        x = self.patch_embed(x)
        _, Hp, Wp, _ = x.shape

        # Reshape: (B, H', W', C) -> (B, H'*W', C)
        x = x.view(B, Hp * Wp, -1)

        # Forward through layers
        # Layer 1
        for blk in self.layers[0]:
            x = blk(x)
        skip = x  # Save skip connection

        # Downsample
        x = self.downsample(x)

        # Layer 2 & 3
        for blk in self.layers[1]:
            x = blk(x)
        for blk in self.layers[2]:
            x = blk(x)

        # Upsample
        x = self.upsample(x)

        # Layer 4
        for blk in self.layers[3]:
            x = blk(x)

        # Skip connection
        x = self.norm(x)
        x = torch.cat([x, skip], dim=-1)

        # Reshape: (B, H'*W', 2C) -> (B, 2C, H', W')
        x = x.transpose(1, 2).reshape(B, -1, Hp, Wp)

        # Patch Recovery: (B, 2C, H', W') -> (B, T_out*C, H, W)
        x = self.patch_recovery(x)

        # Reshape: (B, T_out*C, H, W) -> (B, T_out, C, H, W)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.output_len, c=C)

        return x
