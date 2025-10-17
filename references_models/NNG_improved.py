"""
NNG (Neural Network Weather Prediction with Graph) 模型 - 改进版
基于图神经网络的天气预报模型

主要改进:
1. 使用einops重构张量操作，提高代码可读性
2. 复用Pangu/Fengwu中已改进的模块
3. 添加详细的中文注释
4. 优化代码结构

依赖要求:
- torch
- einops
- dgl (必需): pip install dgl-cu118 (for CUDA 11.8) 或 pip install dgl (for CPU)
"""

import math
import warnings
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
import dgl
from dgl import DGLGraph
from dgl.nn import GraphConv

# ==================== 从Pangu/Fengwu复用的基础模块 ====================

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


# ==================== Mesh和Graph生成模块（使用einops改进） ====================

def generate_latlon_grid(resolution):
    """
    生成经纬度网格坐标

    Args:
        resolution: (lat_size, lon_size) 网格分辨率

    Returns:
        lat_grid, lon_grid: 经纬度网格坐标
    """
    lat_size, lon_size = resolution

    # 生成纬度和经度坐标
    lat = torch.linspace(-90, 90, lat_size)
    # torch.linspace不支持endpoint参数，手动计算不包含端点的范围
    lon = torch.linspace(0, 360 * (lon_size - 1) / lon_size, lon_size)

    # 创建网格
    lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing='xy')

    return lat_grid, lon_grid


def generate_mesh_graph(lat_grid, lon_grid, k_neighbors=8):
    """
    基于经纬度网格生成图结构

    Args:
        lat_grid: 纬度网格
        lon_grid: 经度网格
        k_neighbors: 每个节点的邻居数

    Returns:
        graph: DGL图对象
    """
    # 展平网格坐标
    lat_flat = rearrange(lat_grid, 'h w -> (h w)')
    lon_flat = rearrange(lon_grid, 'h w -> (h w)')

    # 将经纬度转换为笛卡尔坐标（考虑地球球形）
    lat_rad = lat_flat * math.pi / 180
    lon_rad = lon_flat * math.pi / 180

    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)

    # 堆叠为3D坐标
    coords = torch.stack([x, y, z], dim=-1)  # (N, 3)

    # 计算距离矩阵
    dist_matrix = torch.cdist(coords, coords)  # (N, N)

    # 找到k个最近邻
    _, indices = torch.topk(dist_matrix, k=k_neighbors+1, largest=False, dim=1)
    indices = indices[:, 1:]  # 排除自身

    # 构建边
    num_nodes = lat_flat.shape[0]
    src = repeat(torch.arange(num_nodes), 'n -> (n k)', k=k_neighbors)
    dst = rearrange(indices, 'n k -> (n k)')

    # 创建DGL图
    graph = dgl.graph((src, dst))

    # 添加自环以避免0入度节点问题
    graph = dgl.add_self_loop(graph)

    # 添加节点特征（经纬度坐标）
    graph.ndata['lat'] = lat_flat
    graph.ndata['lon'] = lon_flat
    graph.ndata['coords'] = coords

    return graph


# ==================== Graph Encoder模块（使用einops改进） ====================

class GraphEncoder(nn.Module):
    """
    图编码器
    使用图卷积网络处理mesh数据
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=3,
        activation=nn.GELU,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 图卷积层
        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.gnn_layers.append(
                GraphConv(hidden_dim, hidden_dim, activation=activation())
            )
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, features):
        """
        Args:
            graph: DGL图
            features: 节点特征 (num_nodes, input_dim)

        Returns:
            output: 编码后的特征 (num_nodes, output_dim)
        """
        # 输入投影
        x = self.input_proj(features)

        # 图卷积层
        for gnn, norm in zip(self.gnn_layers, self.norm_layers):
            residual = x
            x = gnn(graph, x)
            x = norm(x)
            x = self.dropout(x)
            x = x + residual  # 残差连接

        # 输出投影
        output = self.output_proj(x)

        return output


# ==================== Message Passing模块（使用einops改进） ====================

class MessagePassingLayer(nn.Module):
    """
    消息传递层
    在图上进行特征聚合和更新
    """
    def __init__(
        self,
        node_dim,
        edge_dim=None,
        hidden_dim=None,
        activation=nn.GELU,
        dropout=0.1
    ):
        super().__init__()

        hidden_dim = hidden_dim or node_dim * 2

        # 消息函数
        if edge_dim is not None:
            self.message_func = nn.Sequential(
                nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, node_dim)
            )
        else:
            self.message_func = nn.Sequential(
                nn.Linear(node_dim * 2, hidden_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, node_dim)
            )

        # 更新函数
        self.update_func = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, graph, node_features, edge_features=None):
        """
        Args:
            graph: DGL图
            node_features: 节点特征 (num_nodes, node_dim)
            edge_features: 边特征 (num_edges, edge_dim) 可选

        Returns:
            updated_features: 更新后的节点特征
        """
        with graph.local_scope():
            # 设置节点特征
            graph.ndata['h'] = node_features

            # 消息传递
            if edge_features is not None:
                graph.edata['e'] = edge_features
                graph.apply_edges(self.edge_message_func)
            else:
                graph.apply_edges(self.simple_message_func)

            # 聚合消息
            graph.update_all(
                dgl.function.copy_e('m', 'm'),
                dgl.function.mean('m', 'agg')
            )

            # 更新节点
            agg_features = graph.ndata['agg']
            combined = torch.cat([node_features, agg_features], dim=-1)
            updated = self.update_func(combined)

            # 残差连接和归一化
            output = self.norm(updated + node_features)

            return output

    def edge_message_func(self, edges):
        """计算边消息（包含边特征）"""
        src_h = edges.src['h']
        dst_h = edges.dst['h']
        edge_e = edges.data['e']
        combined = torch.cat([src_h, dst_h, edge_e], dim=-1)
        message = self.message_func(combined)
        return {'m': message}

    def simple_message_func(self, edges):
        """计算边消息（不包含边特征）"""
        src_h = edges.src['h']
        dst_h = edges.dst['h']
        combined = torch.cat([src_h, dst_h], dim=-1)
        message = self.message_func(combined)
        return {'m': message}


# ==================== Processor模块（使用einops改进） ====================

class Processor(nn.Module):
    """
    处理器模块
    多层消息传递和特征更新
    """
    def __init__(
        self,
        node_dim,
        edge_dim=None,
        num_layers=6,
        hidden_dim=None,
        activation=nn.GELU,
        dropout=0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            MessagePassingLayer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, graph, node_features, edge_features=None):
        """
        多层消息传递

        Args:
            graph: DGL图
            node_features: 节点特征
            edge_features: 边特征（可选）

        Returns:
            processed_features: 处理后的节点特征
        """
        x = node_features

        for layer in self.layers:
            x = layer(graph, x, edge_features)

        return x


# ==================== Grid2Mesh和Mesh2Grid模块（使用einops改进） ====================

class Grid2Mesh(nn.Module):
    """
    网格到Mesh的转换
    将规则网格数据映射到图节点
    """
    def __init__(self, grid_dim, mesh_dim, hidden_dim=256):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(grid_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mesh_dim)
        )

    def forward(self, grid_features, graph):
        """
        Args:
            grid_features: 网格特征 (B, C, H, W)
            graph: DGL图对象

        Returns:
            mesh_features: Mesh节点特征 (num_nodes, mesh_dim)
        """
        B, C, H, W = grid_features.shape

        # 重排为 (B, H*W, C)
        features = rearrange(grid_features, 'b c h w -> b (h w) c')

        # 如果batch size > 1，需要处理
        if B > 1:
            # 这里简化处理，实际应该根据graph的batch信息
            features = reduce(features, 'b n c -> n c', 'mean')
        else:
            features = features[0]

        # 投影到mesh维度
        mesh_features = self.projection(features)

        return mesh_features


class Mesh2Grid(nn.Module):
    """
    Mesh到网格的转换
    将图节点特征映射回规则网格
    """
    def __init__(self, mesh_dim, grid_dim, grid_shape, hidden_dim=256):
        super().__init__()

        self.grid_shape = grid_shape  # (H, W)

        self.projection = nn.Sequential(
            nn.Linear(mesh_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, grid_dim)
        )

    def forward(self, mesh_features, batch_size=1):
        """
        Args:
            mesh_features: Mesh节点特征 (num_nodes, mesh_dim)
            batch_size: 批次大小

        Returns:
            grid_features: 网格特征 (B, C, H, W)
        """
        # 投影到网格维度
        features = self.projection(mesh_features)

        # 重排为网格形状
        H, W = self.grid_shape
        C = features.shape[-1]

        # 重排为 (H, W, C)
        features = rearrange(features, '(h w) c -> h w c', h=H, w=W)

        # 转换为 (B, C, H, W)
        features = rearrange(features, 'h w c -> c h w')

        # 添加batch维度
        if batch_size > 1:
            features = repeat(features, 'c h w -> b c h w', b=batch_size)
        else:
            features = features.unsqueeze(0)

        return features


# ==================== 主模型NNG ====================

class NNG(nn.Module):
    """
    NNG (Neural Network Graph) 气象预测模型

    基于图神经网络的天气预报模型，结合了规则网格和图结构的优势：
    1. 使用规则网格输入/输出，便于与传统数值模式对接
    2. 内部使用图结构处理，更好地建模地球球形几何
    3. 多尺度消息传递，捕获不同尺度的大气动力过程

    输入格式: (B, 69, H, W) - 与Fengwu/Pangu相同的输入格式
    输出格式: (B, 69, H, W) - 预测的下一时刻气象场
    """
    def __init__(
        self,
        input_channels=69,
        hidden_channels=256,
        mesh_size=128,
        grid_shape=(120, 240),
        num_mesh_layers=6,
        num_processor_layers=12,
        k_neighbors=8,
        dropout=0.1
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.mesh_size = mesh_size
        self.grid_shape = grid_shape

        # ==================== 网格和图生成 ====================
        # 生成经纬度网格
        lat_grid, lon_grid = generate_latlon_grid(grid_shape)
        self.register_buffer('lat_grid', lat_grid)
        self.register_buffer('lon_grid', lon_grid)

        # 生成mesh图
        self.graph = generate_mesh_graph(lat_grid, lon_grid, k_neighbors)
        # 将graph注册为buffer，这样它会随模型一起移到GPU
        # 注意：DGL图不能直接作为buffer，需要在forward中处理设备转移

        # ==================== 编码器 ====================
        # 输入编码
        self.input_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU()
        )

        # Grid到Mesh转换
        self.grid2mesh = Grid2Mesh(
            grid_dim=hidden_channels,
            mesh_dim=mesh_size,
            hidden_dim=hidden_channels * 2
        )

        # ==================== 图处理器 ====================
        # Mesh编码器
        self.mesh_encoder = GraphEncoder(
            input_dim=mesh_size,
            hidden_dim=mesh_size * 2,
            output_dim=mesh_size,
            num_layers=num_mesh_layers,
            dropout=dropout
        )

        # 处理器（多层消息传递）
        self.processor = Processor(
            node_dim=mesh_size,
            edge_dim=None,
            num_layers=num_processor_layers,
            hidden_dim=mesh_size * 2,
            dropout=dropout
        )

        # Mesh解码器
        self.mesh_decoder = GraphEncoder(
            input_dim=mesh_size,
            hidden_dim=mesh_size * 2,
            output_dim=mesh_size,
            num_layers=num_mesh_layers,
            dropout=dropout
        )

        # ==================== 解码器 ====================
        # Mesh到Grid转换
        self.mesh2grid = Mesh2Grid(
            mesh_dim=mesh_size,
            grid_dim=hidden_channels,
            grid_shape=grid_shape,
            hidden_dim=hidden_channels * 2
        )

        # 输出解码
        self.output_decoder = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=1)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 (B, 69, H, W)

        Returns:
            output: 预测结果 (B, 69, H, W)
        """
        B, C, H, W = x.shape

        # 将图移到与输入相同的设备
        device = x.device
        graph = self.graph.to(device)

        # ==================== 编码阶段 ====================
        # 输入编码: (B, 69, H, W) -> (B, hidden_channels, H, W)
        encoded = self.input_encoder(x)

        # 保存skip连接
        skip = encoded

        # Grid到Mesh转换: (B, hidden_channels, H, W) -> (num_nodes, mesh_size)
        mesh_features = self.grid2mesh(encoded, graph)

        # ==================== 图处理阶段 ====================
        # Mesh编码
        mesh_features = self.mesh_encoder(graph, mesh_features)

        # 消息传递处理
        mesh_features = self.processor(graph, mesh_features)

        # Mesh解码
        mesh_features = self.mesh_decoder(graph, mesh_features)

        # ==================== 解码阶段 ====================
        # Mesh到Grid转换: (num_nodes, mesh_size) -> (B, hidden_channels, H, W)
        grid_features = self.mesh2grid(mesh_features, batch_size=B)

        # 与skip连接合并
        combined = torch.cat([grid_features, skip], dim=1)

        # 输出解码: (B, hidden_channels*2, H, W) -> (B, 69, H, W)
        output = self.output_decoder(combined)

        # 残差连接
        output = output + x

        return output


if __name__ == '__main__':
    """测试代码"""
    print("测试NNG模型...")
    print("注意：需要安装DGL库")
    print("CUDA版本: pip install dgl-cu118")
    print("CPU版本: pip install dgl")
    print()

    try:
        # 创建模型
        model = NNG(
            input_channels=69,
            hidden_channels=256,
            mesh_size=128,
            grid_shape=(120, 240),
            num_mesh_layers=6,
            num_processor_layers=12
        )

        # 创建测试输入
        x = torch.randn(2, 69, 120, 240)

        # 前向传播
        output = model(x)

        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数量: {total_params:,}")

        print("\n✓ NNG模型测试成功！")

    except Exception as e:
        print(f"✗ 错误: {e}")
        print("\n请确保已安装DGL库:")
        print("1. 对于CUDA 11.8: pip install dgl-cu118")
        print("2. 对于CPU: pip install dgl")
