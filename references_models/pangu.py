import math
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np


# ==================== Utility Functions ====================


def get_pad(input_resolution, window_size):
    """Calculate padding needed for window partitioning"""
    padding = []
    for inp, win in zip(reversed(input_resolution), reversed(window_size)):
        remainder = inp % win
        if remainder:
            pad = win - remainder
            padding.extend([pad // 2, pad - pad // 2])
        else:
            padding.extend([0, 0])
    return tuple(padding)


def crop(x, resolution):
    """Center crop to target resolution"""
    *_, *spatial_dims = x.shape
    slices = [slice(None)] * (x.ndim - len(resolution))
    for curr, target in zip(spatial_dims, resolution):
        pad = curr - target
        slices.append(slice(pad // 2, curr - (pad - pad // 2)))
    return x[tuple(slices)]


def get_earth_position_index(window_size):
    """Generate position indices for Earth-aware attention"""
    ndim = len(window_size)
    if ndim == 3:
        coords_q = torch.stack(
            torch.meshgrid([torch.arange(s) for s in window_size], indexing="ij")
        )
        coords_k = torch.stack(
            torch.meshgrid([-torch.arange(s) * s for s in window_size], indexing="ij")
        )
    else:  # ndim == 2
        coords_q = torch.stack(
            torch.meshgrid([torch.arange(s) for s in window_size], indexing="ij")
        )
        coords_k = torch.stack(
            torch.meshgrid([-torch.arange(s) * s for s in window_size], indexing="ij")
        )

    coords = (
        coords_q.flatten(1).unsqueeze(2) - coords_k.flatten(1).unsqueeze(1)
    ).permute(1, 2, 0)

    # Shift indices to start from 0
    if ndim == 3:
        coords[:, :, 2] += window_size[2] - 1
        coords[:, :, 1] *= 2 * window_size[2] - 1
        coords[:, :, 0] *= (2 * window_size[2] - 1) * window_size[1] * window_size[1]
    else:
        coords[:, :, 1] += window_size[1] - 1
        coords[:, :, 0] *= 2 * window_size[1] - 1

    return coords.sum(-1)


def get_shift_window_mask(input_resolution, window_size, shift_size):
    """Generate attention mask for shifted windows"""
    ndim = len(window_size)
    if ndim == 3:
        img_mask = torch.zeros(
            (1, *input_resolution[:2], input_resolution[2] + shift_size[2], 1)
        )
        slices = [
            slice(0, -window_size[i]),
            slice(-window_size[i], -shift_size[i]),
            slice(-shift_size[i], None),
        ]
        cnt = 0
        for pl in slices:
            for lat in slices:
                for lon in slices:
                    img_mask[:, pl, lat, lon, :] = cnt
                    cnt += 1
        img_mask = img_mask[:, :, :, : input_resolution[2], :]
        pattern = "b pl lat lon c -> b npl nlat (wpl wlat wlon)"
    else:
        img_mask = torch.zeros(
            (1, input_resolution[0], input_resolution[1] + shift_size[1], 1)
        )
        slices = [
            slice(0, -window_size[i]),
            slice(-window_size[i], -shift_size[i]),
            slice(-shift_size[i], None),
        ]
        cnt = 0
        for lat in slices:
            for lon in slices:
                img_mask[:, lat, lon, :] = cnt
                cnt += 1
        img_mask = img_mask[:, :, : input_resolution[1], :]
        pattern = "b lat lon c -> b nlat (wlat wlon)"

    if ndim == 3:
        mask_windows = rearrange(
            img_mask,
            "b (npl wpl) (nlat wlat) (nlon wlon) c -> (b nlon) (npl nlat) (wpl wlat wlon) c",
            wpl=window_size[0],
            wlat=window_size[1],
            wlon=window_size[2],
        )
    else:
        mask_windows = rearrange(
            img_mask,
            "b (nlat wlat) (nlon wlon) c -> (b nlon) nlat (wlat wlon) c",
            wlat=window_size[0],
            wlon=window_size[1],
        )

    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0
    )


# ==================== Patch Embedding & Recovery ====================


class PatchEmbed(nn.Module):
    """Unified patch embedding for 2D and 3D inputs"""

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        ndim = len(img_size)
        self.ndim = ndim

        # Calculate padding
        padding = []
        for size, patch in zip(reversed(img_size), reversed(patch_size)):
            remainder = size % patch
            if remainder:
                pad = patch - remainder
                padding.extend([pad // 2, pad - pad // 2])
            else:
                padding.extend([0, 0])

        ConvLayer = nn.Conv3d if ndim == 3 else nn.Conv2d
        PadLayer = nn.ZeroPad3d if ndim == 3 else nn.ZeroPad2d

        self.pad = PadLayer(tuple(padding))
        self.proj = ConvLayer(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.pad(x)
        x = self.proj(x)
        if self.ndim == 3:
            x = rearrange(x, "b c pl lat lon -> b pl lat lon c")
        else:
            x = rearrange(x, "b c lat lon -> b lat lon c")
        x = self.norm(x)
        if self.ndim == 3:
            x = rearrange(x, "b pl lat lon c -> b c pl lat lon")
        else:
            x = rearrange(x, "b lat lon c -> b c lat lon")
        return x


class PatchRecovery(nn.Module):
    """Unified patch recovery for 2D and 3D outputs"""

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        ConvLayer = nn.ConvTranspose3d if len(img_size) == 3 else nn.ConvTranspose2d
        self.conv = ConvLayer(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        x = self.conv(x)
        return crop(x, self.img_size)


# ==================== Up/Down Sampling ====================


class UpSample(nn.Module):
    """Unified upsampling for 2D and 3D features"""

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.ndim = len(input_resolution)

        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.linear1(x)

        if self.ndim == 3:
            pl, lat, lon = self.input_resolution
            x = rearrange(
                x,
                "b (pl lat lon) (h w c) -> b pl (lat h) (lon w) c",
                pl=pl,
                lat=lat,
                lon=lon,
                h=2,
                w=2,
            )
            x = crop(
                rearrange(x, "b pl lat lon c -> b c pl lat lon"), self.output_resolution
            )
            x = rearrange(x, "b c pl lat lon -> b (pl lat lon) c")
        else:
            lat, lon = self.input_resolution
            x = rearrange(
                x,
                "b (lat lon) (h w c) -> b (lat h) (lon w) c",
                lat=lat,
                lon=lon,
                h=2,
                w=2,
            )
            x = crop(rearrange(x, "b lat lon c -> b c lat lon"), self.output_resolution)
            x = rearrange(x, "b c lat lon -> b (lat lon) c")

        return self.linear2(self.norm(x))


class DownSample(nn.Module):
    """Unified downsampling for 2D and 3D features"""

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.ndim = len(input_resolution)

        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)

        # Calculate padding
        padding = []
        for out_res, in_res in zip(
            reversed(output_resolution), reversed(input_resolution)
        ):
            pad = out_res * 2 - in_res
            padding.extend([pad // 2, pad - pad // 2])

        PadLayer = nn.ZeroPad3d if self.ndim == 3 else nn.ZeroPad2d
        self.pad = PadLayer(tuple(padding))

    def forward(self, x):
        B, N, C = x.shape

        if self.ndim == 3:
            pl, lat, lon = self.input_resolution
            opl, olat, olon = self.output_resolution
            x = rearrange(
                x, "b (pl lat lon) c -> b c pl lat lon", pl=pl, lat=lat, lon=lon
            )
            x = self.pad(x)
            x = rearrange(
                x,
                "b c pl (olat h) (olon w) -> b (pl olat olon) (h w c)",
                olat=olat,
                olon=olon,
                h=2,
                w=2,
            )
        else:
            lat, lon = self.input_resolution
            olat, olon = self.output_resolution
            x = rearrange(x, "b (lat lon) c -> b c lat lon", lat=lat, lon=lon)
            x = self.pad(x)
            x = rearrange(
                x,
                "b c (olat h) (olon w) -> b (olat olon) (h w c)",
                olat=olat,
                olon=olon,
                h=2,
                w=2,
            )

        return self.linear(self.norm(x))


# ==================== Attention & Transformer Blocks ====================


class EarthAttention(nn.Module):
    """Earth-specific attention mechanism for 2D/3D data"""

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.ndim = len(window_size)

        # Earth position bias
        if self.ndim == 3:
            self.type_of_windows = (input_resolution[0] // window_size[0]) * (
                input_resolution[1] // window_size[1]
            )
            bias_size = (
                (window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1)
            )
        else:
            self.type_of_windows = input_resolution[0] // window_size[0]
            bias_size = (window_size[0] ** 2) * (window_size[1] * 2 - 1)

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(bias_size, self.type_of_windows, num_heads)
        )
        nn.init.trunc_normal_(self.earth_position_bias_table, std=0.02)

        self.register_buffer(
            "earth_position_index", get_earth_position_index(window_size)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, nW_, N, C = x.shape

        qkv = rearrange(
            self.qkv(x),
            "b nw n (three h d) -> three b h nw n d",
            three=3,
            h=self.num_heads,
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)

        # Add earth position bias
        N_win = math.prod(self.window_size)
        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(N_win, N_win, self.type_of_windows, -1)
        attn = attn + rearrange(
            earth_position_bias, "n1 n2 nw h -> h nw n1 n2"
        ).unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            nLon = mask.shape[0]
            attn = rearrange(
                attn, "(b nlon) h nw n1 n2 -> b nlon h nw n1 n2", nlon=nLon
            )
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = rearrange(attn, "b nlon h nw n1 n2 -> (b nlon) h nw n1 n2")

        attn = self.attn_drop(self.softmax(attn))
        x = rearrange(attn @ v, "b h nw n d -> b nw n (h d)")

        return self.proj_drop(self.proj(x))


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """Unified Transformer block for 2D/3D data"""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size,
        shift_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size or tuple([0] * len(window_size))
        self.ndim = len(window_size)

        self.norm1 = nn.LayerNorm(dim)

        # Padding
        padding = get_pad(input_resolution, window_size)
        PadLayer = nn.ZeroPad3d if self.ndim == 3 else nn.ZeroPad2d
        self.pad = PadLayer(padding)

        pad_resolution = list(input_resolution)
        if self.ndim == 3:
            pad_resolution[0] += padding[-1] + padding[-2]
            pad_resolution[1] += padding[2] + padding[3]
            pad_resolution[2] += padding[0] + padding[1]
        else:
            pad_resolution[0] += padding[2] + padding[3]
            pad_resolution[1] += padding[0] + padding[1]

        self.attn = EarthAttention(
            dim, pad_resolution, window_size, num_heads, qkv_bias, attn_drop, drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

        self.roll = any(s > 0 for s in self.shift_size)
        if self.roll:
            self.register_buffer(
                "attn_mask",
                get_shift_window_mask(pad_resolution, window_size, self.shift_size),
            )
        else:
            self.attn_mask = None

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # Reshape to spatial
        if self.ndim == 3:
            pl, lat, lon = self.input_resolution
            x = rearrange(
                x, "b (pl lat lon) c -> b c pl lat lon", pl=pl, lat=lat, lon=lon
            )
            x = self.pad(x)
            _, _, pl_p, lat_p, lon_p = x.shape
            x = rearrange(x, "b c pl lat lon -> b pl lat lon c")

            # Shifted window partition
            if self.roll:
                x = torch.roll(
                    x, shifts=tuple(-s for s in self.shift_size), dims=(1, 2, 3)
                )

            x = rearrange(
                x,
                "b (npl wpl) (nlat wlat) (nlon wlon) c -> (b nlon) (npl nlat) (wpl wlat wlon) c",
                wpl=self.window_size[0],
                wlat=self.window_size[1],
                wlon=self.window_size[2],
            )

            # Attention
            x = self.attn(x, self.attn_mask)

            # Reverse
            x = rearrange(
                x,
                "(b nlon) (npl nlat) (wpl wlat wlon) c -> b (npl wpl) (nlat wlat) (nlon wlon) c",
                b=shortcut.shape[0],
                nlon=lon_p // self.window_size[2],
                npl=pl_p // self.window_size[0],
                nlat=lat_p // self.window_size[1],
                wpl=self.window_size[0],
                wlat=self.window_size[1],
                wlon=self.window_size[2],
            )

            if self.roll:
                x = torch.roll(x, shifts=self.shift_size, dims=(1, 2, 3))

            x = rearrange(x, "b pl lat lon c -> b c pl lat lon")
            x = crop(x, self.input_resolution)
            x = rearrange(x, "b c pl lat lon -> b (pl lat lon) c")
        else:
            lat, lon = self.input_resolution
            x = rearrange(x, "b (lat lon) c -> b c lat lon", lat=lat, lon=lon)
            x = self.pad(x)
            _, _, lat_p, lon_p = x.shape
            x = rearrange(x, "b c lat lon -> b lat lon c")

            if self.roll:
                x = torch.roll(
                    x, shifts=tuple(-s for s in self.shift_size), dims=(1, 2)
                )

            x = rearrange(
                x,
                "b (nlat wlat) (nlon wlon) c -> (b nlon) nlat (wlat wlon) c",
                wlat=self.window_size[0],
                wlon=self.window_size[1],
            )

            x = self.attn(x, self.attn_mask)

            x = rearrange(
                x,
                "(b nlon) nlat (wlat wlon) c -> b (nlat wlat) (nlon wlon) c",
                b=shortcut.shape[0],
                nlon=lon_p // self.window_size[1],
                nlat=lat_p // self.window_size[0],
                wlat=self.window_size[0],
                wlon=self.window_size[1],
            )

            if self.roll:
                x = torch.roll(x, shifts=self.shift_size, dims=(1, 2))

            x = rearrange(x, "b lat lon c -> b c lat lon")
            x = crop(x, self.input_resolution)
            x = rearrange(x, "b c lat lon -> b (lat lon) c")

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FuserLayer(nn.Module):
    """Stack of Transformer blocks with alternating window shifts"""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        drop_path = (
            drop_path if isinstance(drop_path, (list, tuple)) else [drop_path] * depth
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=tuple([0] * len(window_size))
                    if i % 2 == 0
                    else tuple([s // 2 for s in window_size]),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ==================== Pangu Weather Model ====================


class Pangu(nn.Module):
    """Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast"""

    def __init__(
        self,
        img_size=(120, 240),
        patch_size=(2, 4, 4),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
    ):
        super().__init__()

        # Patch embedding
        self.patchembed2d = PatchEmbed(img_size, patch_size[1:], 4, embed_dim)
        self.patchembed3d = PatchEmbed((13, *img_size), patch_size, 5, embed_dim)

        # Calculate resolutions
        res = tuple(math.ceil(img_size[i] / patch_size[i + 1]) for i in range(2))
        patched_inp_shape = (8, *res)
        patched_down_shape = (8, math.ceil(res[0] / 2), math.ceil(res[1] / 2))

        drop_path = np.linspace(0, 0.2, 8).tolist()

        # Encoder
        self.layer1 = FuserLayer(
            embed_dim,
            patched_inp_shape,
            2,
            num_heads[0],
            window_size,
            drop_path=drop_path[:2],
        )
        self.downsample = DownSample(embed_dim, patched_inp_shape, patched_down_shape)

        # Bottleneck
        self.layer2 = FuserLayer(
            embed_dim * 2,
            patched_down_shape,
            6,
            num_heads[1],
            window_size,
            drop_path=drop_path[2:],
        )
        self.layer3 = FuserLayer(
            embed_dim * 2,
            patched_down_shape,
            6,
            num_heads[2],
            window_size,
            drop_path=drop_path[2:],
        )

        # Decoder
        self.upsample = UpSample(
            embed_dim * 2, embed_dim, patched_down_shape, patched_inp_shape
        )
        self.layer4 = FuserLayer(
            embed_dim,
            patched_inp_shape,
            2,
            num_heads[3],
            window_size,
            drop_path=drop_path[:2],
        )

        # Patch recovery
        self.patchrecovery2d = PatchRecovery(img_size, patch_size[1:], 2 * embed_dim, 4)
        self.patchrecovery3d = PatchRecovery(
            (13, *img_size), patch_size, 2 * embed_dim, 5
        )

    def forward(self, x):
        """
        Args:
            x: (B, 69, lat, lon) where 69 = 5*13 (upper air) + 4 (surface)
        Returns:
            (B, 69, lat, lon) prediction
        """
        # Split input
        surface = x[:, 65:69]
        upper_air = rearrange(
            x[:, :65], "b (c pl) lat lon -> b c pl lat lon", c=5, pl=13
        )

        # Patch embedding
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        # Concatenate along pressure level dimension
        x = torch.cat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = rearrange(x, "b c pl lat lon -> b (pl lat lon) c")

        # Encoder
        x = self.layer1(x)
        skip = x
        x = self.downsample(x)

        # Bottleneck
        x = self.layer2(x)
        x = self.layer3(x)

        # Decoder
        x = self.upsample(x)
        x = self.layer4(x)

        # Skip connection and reshape
        output = torch.cat([x, skip], dim=-1)
        output = rearrange(
            output, "b (pl lat lon) c -> b c pl lat lon", pl=Pl, lat=Lat, lon=Lon
        )

        # Patch recovery
        output_surface = self.patchrecovery2d(output[:, :, 0])
        output_upper_air = self.patchrecovery3d(output[:, :, 1:])
        output_upper_air = rearrange(
            output_upper_air, "b c pl lat lon -> b (c pl) lat lon"
        )

        return torch.cat([output_upper_air, output_surface], dim=1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Pangu().to(device)

    # Test
    x = torch.randn(1, 69, 120, 240).to(device)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
