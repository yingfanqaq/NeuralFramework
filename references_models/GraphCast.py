"""
Simplified GraphCast implementation with einops for better readability.
Removes distributed training, checkpoint complexity, and unused features.
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple, List, Union
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import transform

import dgl
import dgl.function as fn
from dgl import DGLGraph

# =============================================================================
# Geometry Utilities
# =============================================================================


def deg2rad(deg: torch.Tensor) -> torch.Tensor:
    return deg * np.pi / 180


def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    return rad * 180 / np.pi


def latlon2xyz(
    latlon: torch.Tensor, radius: float = 1, unit: str = "deg"
) -> torch.Tensor:
    """Convert lat/lon to 3D cartesian coordinates."""
    if unit == "deg":
        latlon = deg2rad(latlon)
    lat, lon = latlon[..., 0], latlon[..., 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


def xyz2latlon(xyz: torch.Tensor, radius: float = 1, unit: str = "deg") -> torch.Tensor:
    """Convert 3D cartesian to lat/lon coordinates."""
    lat = torch.arcsin(xyz[..., 2] / radius)
    lon = torch.arctan2(xyz[..., 1], xyz[..., 0])
    result = torch.stack([lat, lon], dim=-1)
    return rad2deg(result) if unit == "deg" else result


def geospatial_rotation(
    invar: torch.Tensor, theta: torch.Tensor, axis: str, unit: str = "rad"
) -> torch.Tensor:
    """Rotate coordinates around specified axis."""
    if unit == "deg":
        theta = deg2rad(theta)

    invar = rearrange(invar, "... d -> ... d 1")
    rotation = torch.zeros((*theta.shape, 3, 3), device=theta.device)
    cos, sin = torch.cos(theta), torch.sin(theta)

    if axis == "x":
        rotation[..., 0, 0] = 1.0
        rotation[..., 1, 1] = cos
        rotation[..., 1, 2] = -sin
        rotation[..., 2, 1] = sin
        rotation[..., 2, 2] = cos
    elif axis == "y":
        rotation[..., 0, 0] = cos
        rotation[..., 0, 2] = sin
        rotation[..., 1, 1] = 1.0
        rotation[..., 2, 0] = -sin
        rotation[..., 2, 2] = cos
    elif axis == "z":
        rotation[..., 0, 0] = cos
        rotation[..., 0, 1] = -sin
        rotation[..., 1, 0] = sin
        rotation[..., 1, 1] = cos
        rotation[..., 2, 2] = 1.0

    outvar = torch.matmul(rotation, invar)
    return rearrange(outvar, "... d 1 -> ... d")


# =============================================================================
# Mesh Generation
# =============================================================================


class TriangularMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = vertices
        self.faces = faces


def get_icosahedron() -> TriangularMesh:
    """Generate icosahedron mesh."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for c1 in [1.0, -1.0]:
        for c2 in [phi, -phi]:
            vertices.extend([(c1, c2, 0.0), (0.0, c1, c2), (c2, 0.0, c1)])

    vertices = np.array(vertices, dtype=np.float32)
    vertices /= np.linalg.norm([1.0, phi])

    faces = [
        (0, 1, 2),
        (0, 6, 1),
        (8, 0, 2),
        (8, 4, 0),
        (3, 8, 2),
        (3, 2, 7),
        (7, 2, 1),
        (0, 4, 6),
        (4, 11, 6),
        (6, 11, 5),
        (1, 5, 7),
        (4, 10, 11),
        (4, 8, 10),
        (10, 8, 3),
        (10, 3, 9),
        (11, 10, 9),
        (11, 9, 5),
        (5, 9, 7),
        (9, 3, 7),
        (1, 6, 5),
    ]

    # Rotation for proper alignment
    angle = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle) / 2
    rotation = transform.Rotation.from_euler(seq="y", angles=rotation_angle)
    vertices = vertices @ rotation.as_matrix().T

    return TriangularMesh(vertices.astype(np.float32), np.array(faces, dtype=np.int32))


def split_mesh(mesh: TriangularMesh) -> TriangularMesh:
    """Split each triangle into 4 smaller triangles."""
    vertex_cache = {}
    all_vertices = list(mesh.vertices)

    def get_midpoint(i1: int, i2: int) -> int:
        key = tuple(sorted([i1, i2]))
        if key not in vertex_cache:
            midpoint = (mesh.vertices[i1] + mesh.vertices[i2]) / 2
            midpoint /= np.linalg.norm(midpoint)
            vertex_cache[key] = len(all_vertices)
            all_vertices.append(midpoint)
        return vertex_cache[key]

    new_faces = []
    for i1, i2, i3 in mesh.faces:
        i12 = get_midpoint(i1, i2)
        i23 = get_midpoint(i2, i3)
        i31 = get_midpoint(i3, i1)
        new_faces.extend(
            [[i1, i12, i31], [i12, i2, i23], [i31, i23, i3], [i12, i23, i31]]
        )

    return TriangularMesh(
        np.array(all_vertices, dtype=np.float32), np.array(new_faces, dtype=np.int32)
    )


def get_mesh_hierarchy(splits: int) -> List[TriangularMesh]:
    """Generate hierarchy of meshes."""
    meshes = [get_icosahedron()]
    for _ in range(splits):
        meshes.append(split_mesh(meshes[-1]))
    return meshes


def merge_meshes(meshes: List[TriangularMesh]) -> TriangularMesh:
    """Merge multiple meshes into one."""
    return TriangularMesh(
        vertices=meshes[-1].vertices,
        faces=np.concatenate([m.faces for m in meshes], axis=0),
    )


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert face indices to edge indices."""
    src = np.concatenate([faces[:, i] for i in range(3)])
    dst = np.concatenate([faces[:, (i + 1) % 3] for i in range(3)])
    return src, dst


# =============================================================================
# Graph Construction
# =============================================================================


def create_graph(src: List, dst: List, bidirected: bool = True) -> DGLGraph:
    """Create DGL graph from edge lists."""
    graph = dgl.graph((src, dst), idtype=torch.int32)
    if bidirected:
        graph = dgl.to_bidirected(graph)
    return graph


def add_edge_features(
    graph: DGLGraph, src_pos: torch.Tensor, dst_pos: torch.Tensor
) -> DGLGraph:
    """Add edge features based on node positions."""
    src_idx, dst_idx = graph.edges()
    src_coords = src_pos[src_idx.long()]
    dst_coords = dst_pos[dst_idx.long()]

    # Convert to local coordinate system
    dst_latlon = xyz2latlon(dst_coords, unit="rad")
    theta_az = torch.where(
        dst_latlon[:, 1] >= 0, 2 * np.pi - dst_latlon[:, 1], -dst_latlon[:, 1]
    )
    theta_polar = torch.where(
        dst_latlon[:, 0] >= 0, dst_latlon[:, 0], 2 * np.pi + dst_latlon[:, 0]
    )

    src_rot = geospatial_rotation(src_coords, theta_az, "z", "rad")
    src_rot = geospatial_rotation(src_rot, theta_polar, "y", "rad")

    # Edge features: displacement + distance
    disp = src_rot - torch.tensor([1, 0, 0], device=src_rot.device)
    dist = torch.norm(disp, dim=-1, keepdim=True)
    max_dist = dist.max()

    graph.edata["x"] = torch.cat([disp / max_dist, dist / max_dist], dim=-1)
    return graph


# =============================================================================
# Graph Operations
# =============================================================================


def concat_efeat_dgl(
    efeat: torch.Tensor, src_feat: torch.Tensor, dst_feat: torch.Tensor, graph: DGLGraph
) -> torch.Tensor:
    """Concatenate edge and node features using DGL."""
    with graph.local_scope():
        graph.srcdata["x"] = src_feat
        graph.dstdata["x"] = dst_feat
        graph.edata["x"] = efeat
        graph.apply_edges(
            lambda edges: {
                "cat": torch.cat(
                    [edges.data["x"], edges.src["x"], edges.dst["x"]], dim=-1
                )
            }
        )
        return graph.edata["cat"]


def aggregate_and_concat_dgl(
    efeat: torch.Tensor, dst_feat: torch.Tensor, graph: DGLGraph, agg: str = "sum"
) -> torch.Tensor:
    """Aggregate edge features to nodes and concatenate."""
    with graph.local_scope():
        graph.edata["x"] = efeat
        agg_fn = fn.sum("m", "h") if agg == "sum" else fn.mean("m", "h")
        graph.update_all(fn.copy_e("x", "m"), agg_fn)
        return torch.cat([graph.dstdata["h"], dst_feat], dim=-1)


# =============================================================================
# Neural Network Modules
# =============================================================================


class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int = 1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.extend([nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeMLP(nn.Module):
    """MLP for edge updates with node feature concatenation."""

    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_layers: int = 1,
    ):
        super().__init__()
        self.mlp = MLP(edge_dim + 2 * node_dim, out_dim, hidden_dim, n_layers)

    def forward(
        self,
        efeat: torch.Tensor,
        src_feat: torch.Tensor,
        dst_feat: torch.Tensor,
        graph: DGLGraph,
    ) -> torch.Tensor:
        cat_feat = concat_efeat_dgl(efeat, src_feat, dst_feat, graph)
        return self.mlp(cat_feat)


class Encoder(nn.Module):
    """Grid to mesh encoder."""

    def __init__(self, hidden_dim: int, n_layers: int, agg: str = "sum"):
        super().__init__()
        self.edge_mlp = EdgeMLP(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, n_layers
        )
        self.src_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.dst_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.agg = agg

    def forward(
        self,
        g2m_efeat: torch.Tensor,
        grid_feat: torch.Tensor,
        mesh_feat: torch.Tensor,
        graph: DGLGraph,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        efeat = self.edge_mlp(g2m_efeat, grid_feat, mesh_feat, graph)
        agg_feat = aggregate_and_concat_dgl(efeat, mesh_feat, graph, self.agg)
        mesh_feat_new = mesh_feat + self.dst_mlp(agg_feat)
        grid_feat_new = grid_feat + self.src_mlp(grid_feat)
        return grid_feat_new, mesh_feat_new


class Decoder(nn.Module):
    """Mesh to grid decoder."""

    def __init__(self, hidden_dim: int, n_layers: int, agg: str = "sum"):
        super().__init__()
        self.edge_mlp = EdgeMLP(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, n_layers
        )
        self.node_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.agg = agg

    def forward(
        self,
        m2g_efeat: torch.Tensor,
        grid_feat: torch.Tensor,
        mesh_feat: torch.Tensor,
        graph: DGLGraph,
    ) -> torch.Tensor:
        efeat = self.edge_mlp(m2g_efeat, mesh_feat, grid_feat, graph)
        agg_feat = aggregate_and_concat_dgl(efeat, grid_feat, graph, self.agg)
        return grid_feat + self.node_mlp(agg_feat)


class ProcessorLayer(nn.Module):
    """Single processor layer for mesh."""

    def __init__(self, hidden_dim: int, n_layers: int, agg: str = "sum"):
        super().__init__()
        self.edge_mlp = EdgeMLP(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, n_layers
        )
        self.node_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.agg = agg

    def forward(
        self, efeat: torch.Tensor, nfeat: torch.Tensor, graph: DGLGraph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        efeat_new = efeat + self.edge_mlp(efeat, nfeat, nfeat, graph)
        agg_feat = aggregate_and_concat_dgl(efeat_new, nfeat, graph, self.agg)
        nfeat_new = nfeat + self.node_mlp(agg_feat)
        return efeat_new, nfeat_new


class Processor(nn.Module):
    """Mesh graph processor with multiple layers."""

    def __init__(
        self, hidden_dim: int, n_layers: int, mlp_layers: int, agg: str = "sum"
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [ProcessorLayer(hidden_dim, mlp_layers, agg) for _ in range(n_layers)]
        )

    def forward(
        self, efeat: torch.Tensor, nfeat: torch.Tensor, graph: DGLGraph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            efeat, nfeat = layer(efeat, nfeat, graph)
        return efeat, nfeat


# =============================================================================
# Graph Builder
# =============================================================================


class GraphBuilder:
    """Build graphs for GraphCast."""

    def __init__(
        self, lat_lon_grid: torch.Tensor, mesh_level: int = 5, multimesh: bool = True
    ):
        self.lat_lon_flat = rearrange(lat_lon_grid, "h w c -> (h w) c")

        # Build mesh hierarchy
        meshes = get_mesh_hierarchy(mesh_level)
        finest = meshes[-1]
        mesh = merge_meshes(meshes) if multimesh else finest

        self.mesh_vertices = torch.from_numpy(mesh.vertices).float()
        mesh_src, mesh_dst = faces_to_edges(mesh.faces)

        # Create mesh graph
        self.mesh_graph = create_graph(mesh_src, mesh_dst)
        self.mesh_graph = add_edge_features(
            self.mesh_graph, self.mesh_vertices, self.mesh_vertices
        )
        self.mesh_graph.ndata["pos"] = self.mesh_vertices

        # Create g2m graph (grid to mesh)
        grid_xyz = latlon2xyz(self.lat_lon_flat)
        max_edge_len = self._get_max_edge_len(finest)
        self.g2m_graph = self._create_bipartite_graph(
            grid_xyz, self.mesh_vertices, max_edge_len * 0.6, is_g2m=True
        )

        # Create m2g graph (mesh to grid)
        centroids = self._get_centroids(mesh)
        self.m2g_graph = self._create_m2g_graph(grid_xyz, centroids, mesh.faces)

    def _get_max_edge_len(self, mesh: TriangularMesh) -> float:
        src, dst = faces_to_edges(mesh.faces)
        diffs = mesh.vertices[src] - mesh.vertices[dst]
        return np.sqrt(np.max(np.sum(diffs**2, axis=1)))

    def _get_centroids(self, mesh: TriangularMesh) -> np.ndarray:
        return np.mean(mesh.vertices[mesh.faces], axis=1)

    def _create_bipartite_graph(
        self,
        src_pos: torch.Tensor,
        dst_pos: Union[np.ndarray, torch.Tensor],
        max_dist: float,
        is_g2m: bool,
    ) -> DGLGraph:
        # Convert to numpy for sklearn if needed
        dst_pos_np = (
            dst_pos.cpu().numpy() if isinstance(dst_pos, torch.Tensor) else dst_pos
        )

        nbrs = NearestNeighbors(n_neighbors=4).fit(dst_pos_np)
        distances, indices = nbrs.kneighbors(src_pos.cpu().numpy())

        src_idx, dst_idx = [], []
        for i in range(len(src_pos)):
            for j in range(4):
                if distances[i][j] <= max_dist:
                    src_idx.append(i)
                    dst_idx.append(indices[i][j])

        graph = dgl.heterograph(
            {("src", "edge", "dst"): (src_idx, dst_idx)}, idtype=torch.int32
        )
        graph.srcdata["pos"] = src_pos.float()

        # Handle both tensor and numpy array
        if isinstance(dst_pos, torch.Tensor):
            graph.dstdata["pos"] = dst_pos.float()
        else:
            graph.dstdata["pos"] = torch.from_numpy(dst_pos).float()

        graph = add_edge_features(graph, graph.srcdata["pos"], graph.dstdata["pos"])
        return graph

    def _create_m2g_graph(
        self, grid_xyz: torch.Tensor, centroids: np.ndarray, faces: np.ndarray
    ) -> DGLGraph:
        nbrs = NearestNeighbors(n_neighbors=1).fit(centroids)
        _, indices = nbrs.kneighbors(grid_xyz.cpu().numpy())
        indices = indices.flatten()

        src_idx = [p for i in indices for p in faces[i]]
        dst_idx = [i for i in range(len(grid_xyz)) for _ in range(3)]

        graph = dgl.heterograph(
            {("src", "edge", "dst"): (src_idx, dst_idx)}, idtype=torch.int32
        )
        graph.srcdata["pos"] = self.mesh_vertices
        graph.dstdata["pos"] = grid_xyz.float()
        graph = add_edge_features(graph, graph.srcdata["pos"], graph.dstdata["pos"])
        return graph


# =============================================================================
# Main GraphCast Model
# =============================================================================


class GraphCast(nn.Module):
    """Simplified GraphCast model."""

    def __init__(
        self,
        input_res: Tuple[int, int] = (120, 240),
        in_channels: int = 69,
        out_channels: int = 69,
        hidden_dim: int = 512,
        mesh_level: int = 5,
        processor_layers: int = 16,
        mlp_layers: int = 1,
        multimesh: bool = True,
        aggregation: str = "sum",
    ):
        super().__init__()
        self.input_res = input_res
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build graphs
        lat = torch.linspace(-90, 90, input_res[0])
        lon = torch.linspace(-180, 180, input_res[1] + 1)[1:]
        lat_lon_grid = torch.stack(torch.meshgrid(lat, lon, indexing="ij"), dim=-1)

        self.graph_builder = GraphBuilder(lat_lon_grid, mesh_level, multimesh)

        # Feature embedders
        self.grid_embed = MLP(in_channels, hidden_dim, hidden_dim, mlp_layers)
        self.mesh_node_embed = MLP(3, hidden_dim, hidden_dim, mlp_layers)  # xyz coords
        self.g2m_edge_embed = MLP(4, hidden_dim, hidden_dim, mlp_layers)
        self.mesh_edge_embed = MLP(4, hidden_dim, hidden_dim, mlp_layers)
        self.m2g_edge_embed = MLP(4, hidden_dim, hidden_dim, mlp_layers)

        # Encoder, processor, decoder
        self.encoder = Encoder(hidden_dim, mlp_layers, aggregation)
        self.processor = Processor(
            hidden_dim, processor_layers, mlp_layers, aggregation
        )
        self.decoder = Decoder(hidden_dim, mlp_layers, aggregation)

        # Output head
        self.output_head = nn.Sequential(
            MLP(hidden_dim, hidden_dim, hidden_dim, mlp_layers),
            nn.Linear(hidden_dim, out_channels),
        )

        # Cache graph features
        self.register_buffer("mesh_pos", self.graph_builder.mesh_vertices)
        self.register_buffer("g2m_efeat", self.graph_builder.g2m_graph.edata["x"])
        self.register_buffer("mesh_efeat", self.graph_builder.mesh_graph.edata["x"])
        self.register_buffer("m2g_efeat", self.graph_builder.m2g_graph.edata["x"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input features
        Returns:
            [B, C_out, H, W] output features
        """
        B = x.shape[0]
        if B != 1:
            raise ValueError("Only batch size 1 supported")

        # Reshape: [B, C, H, W] -> [N, C]
        grid_feat = rearrange(x[0], "c h w -> (h w) c")

        # Embed features
        grid_feat = self.grid_embed(grid_feat)
        mesh_node_feat = self.mesh_node_embed(self.mesh_pos)
        g2m_efeat = self.g2m_edge_embed(self.g2m_efeat)
        mesh_efeat = self.mesh_edge_embed(self.mesh_efeat)
        m2g_efeat = self.m2g_edge_embed(self.m2g_efeat)

        # Encode: grid -> mesh
        grid_feat, mesh_node_feat = self.encoder(
            g2m_efeat, grid_feat, mesh_node_feat, self.graph_builder.g2m_graph
        )

        # Process on mesh
        mesh_efeat, mesh_node_feat = self.processor(
            mesh_efeat, mesh_node_feat, self.graph_builder.mesh_graph
        )

        # Decode: mesh -> grid
        grid_feat = self.decoder(
            m2g_efeat, grid_feat, mesh_node_feat, self.graph_builder.m2g_graph
        )

        # Output
        output = self.output_head(grid_feat)

        # Reshape: [N, C] -> [B, C, H, W]
        output = rearrange(
            output, "(h w) c -> 1 c h w", h=self.input_res[0], w=self.input_res[1]
        )

        return output

    def to(self, *args, **kwargs):
        """Override to() to also move graph structures to the target device."""
        # Call parent to() first - moves parameters and registered buffers
        self = super().to(*args, **kwargs)

        # Parse device from args
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None:
            # Move all graphs to the same device
            self.graph_builder.mesh_graph = self.graph_builder.mesh_graph.to(device)
            self.graph_builder.g2m_graph = self.graph_builder.g2m_graph.to(device)
            self.graph_builder.m2g_graph = self.graph_builder.m2g_graph.to(device)

            # Move mesh_vertices tensor (used in graph construction)
            self.graph_builder.mesh_vertices = self.graph_builder.mesh_vertices.to(
                device
            )

        return self


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 使用较小配置避免显存不足（OOM）
    # 如果有大显存，可以增加 hidden_dim 和 processor_layers
    model = GraphCast(
        input_res=(120, 240),
        in_channels=69,
        out_channels=69,
        hidden_dim=256,  # 减小以节省显存（512 需要 ~10GB）
        mesh_level=4,  # 减小网格层级（5 需要更多显存）
        processor_layers=4,  # 减少层数（16 需要更多显存）
        mlp_layers=1,
    ).to(device)

    x = torch.randn(1, 69, 120, 240).to(device)

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 前向传播
    y = model(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {y.shape}")
    print(f"✅ Test passed!")

    # 显示显存使用（如果是 CUDA）
    if torch.cuda.is_available():
        print(f"📊 GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

