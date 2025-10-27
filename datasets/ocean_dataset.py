import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import os
import pickle
from einops import rearrange, repeat
from .base import MyDataset, MyBase


class OceanDataset(MyDataset):
    def __init__(
        self,
        data_args,
        mode='train',
        **kwargs
    ):
        self.logger = logging.getLogger(__name__)
        self.input_len = data_args.get('input_len', 10)
        self.output_len = data_args.get('output_len', 5)
        self.seq_len = self.input_len + self.output_len
        
        # 'standardize', 'normalize', 'none'
        self.normalization_mode = data_args.get('normalization_mode', 'standardize')
        
        self.patches_per_day = data_args.get('patches_per_day', None)
        self.saving_path = data_args.get('saving_path', None)  # For saving normalization params
        self.mode = mode  # 'train' or 'test'

        data_dir = data_args.get('data_path', '')
        train_ratio = data_args.get('train_ratio', 0.6)
        valid_ratio = data_args.get('valid_ratio', 0.2)
        test_ratio = data_args.get('test_ratio', 0.2)
        
        # Store mask and coordinates for later use
        self.mask = None  # Will store as (patches_per_day, H, W) or (H, W)
        self.lat = None
        self.lon = None
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None
        self.data_indices = None  # Store train/valid/test split indices
        self.mask_per_region = None  # Store mask for each region if available
        
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        
        # Get use_old_split option (default: False)
        # If True: try to load saved indices from data_info.pkl (if exists)
        # If False: always generate new indices
        use_old_split = data_args.get('use_old_split', False)
        
        # Try to load saved data_info if use_old_split=True and saving_path exists
        if use_old_split and self.saving_path is not None:
            try:
                data_info = self.load_data_info(self.saving_path)
                self.data_indices = data_info.get('data_indices', None)
                if self.data_indices is not None:
                    self.logger.info(f"✓ Loaded saved data split from {self.saving_path}/data_info.pkl")
                    self.logger.info(f"  Train: {len(self.data_indices['train'])}, "
                                   f"Valid: {len(self.data_indices['valid'])}, "
                                   f"Test: {len(self.data_indices['test'])}")
                else:
                    self.logger.info(f"✓ data_indices not found in data_info.pkl, will generate new split")
            except FileNotFoundError:
                self.logger.info(f"✓ data_info.pkl not found in {self.saving_path}, will generate new split")
                self.data_indices = None
            except Exception as e:
                self.logger.warning(f"⚠ Failed to load data_info.pkl: {e}, will generate new split")
                self.data_indices = None
        elif use_old_split and self.saving_path is None:
            self.logger.warning(f"⚠ use_old_split=True but saving_path is None, will generate new split")
            self.data_indices = None
        else:
            # use_old_split=False, generate new indices
            self.data_indices = None
        
        train_batchsize = data_args.get('train_batchsize', 32)
        eval_batchsize = data_args.get('eval_batchsize', 32)
        subset = data_args.get('subset', False)
        sub_ratio = data_args.get('subset_ratio', 0.2)
        num_workers = data_args.get('num_workers', 0)
        pin_memory = data_args.get('pin_memory', False)
        
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_batchsize = train_batchsize
        self.eval_batchsize = eval_batchsize

        # Load data (this will populate self.patch_indices and self.data_indices)
        X, y = self.load_data(data_dir, subset, sub_ratio)

        # Create datasets using custom indices (already computed in load_data)
        # Use data_indices to split X, y, and patch_indices
        train_indices = self.data_indices['train']
        valid_indices = self.data_indices['valid']
        test_indices = self.data_indices['test']

        self.train_data = OceanSequenceDataset(
            X[train_indices],
            y[train_indices],
            self.patch_indices[train_indices] if hasattr(self, 'patch_indices') else None
        )
        self.valid_data = OceanSequenceDataset(
            X[valid_indices],
            y[valid_indices],
            self.patch_indices[valid_indices] if hasattr(self, 'patch_indices') else None
        )
        self.test_data = OceanSequenceDataset(
            X[test_indices],
            y[test_indices],
            self.patch_indices[test_indices] if hasattr(self, 'patch_indices') else None
        )

        # Store reference to this dataset for accessing mask_per_region and other attributes
        self.train_data.dataset = self
        self.valid_data.dataset = self
        self.test_data.dataset = self

    @property
    def train_loader(self):
        return DataLoader(self.train_data, batch_size=self.train_batchsize, shuffle=True,
                         num_workers=self.num_workers, pin_memory=self.pin_memory)

    @property
    def valid_loader(self):
        return DataLoader(self.valid_data, batch_size=self.eval_batchsize, shuffle=False,
                         num_workers=self.num_workers, pin_memory=self.pin_memory)

    @property
    def test_loader(self):
        return DataLoader(self.test_data, batch_size=self.eval_batchsize, shuffle=False,
                         num_workers=self.num_workers, pin_memory=self.pin_memory)

    def load_data(self, data_dir, subset, sub_ratio):
        """Load and process ocean data"""
        data_path = data_dir

        self.logger.info(f"Loading ocean data from {data_path}")

        if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
            data, mask, lat, lon = self.load_h5_data(data_path)
        elif data_path.endswith('.mat'):
            data, mask, lat, lon = self.load_mat_data(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}. Supported formats: .h5, .hdf5, .mat")

        if self.patches_per_day is None:
            raise ValueError("patches_per_day must be specified in config! ")

        self.logger.info(f"Using patches_per_day: {self.patches_per_day}")
        self.logger.info(f"Data shape: {data.shape}, Mask shape: {mask.shape}")

        if mask.shape[0] == self.patches_per_day:
            # mask is (patches_per_day, H, W) - one mask per spatial region
            self.mask_per_region = mask  # Store all masks (patches_per_day, H, W)
            self.mask = mask  # Keep for backward compatibility
            self.lat = lat  # (patches_per_day, H, W)
            self.lon = lon  # (patches_per_day, H, W)
            self.logger.info(f"Mask format: per-region (patches_per_day={self.patches_per_day}, H={mask.shape[1]}, W={mask.shape[2]})")
        elif mask.shape[0] == data.shape[0]:
            # mask is (N, H, W) - one mask per timestep per region
            n_days = data.shape[0] // self.patches_per_day
            mask_reshaped = rearrange(mask[:n_days * self.patches_per_day],
                                     '(d p) h w -> d p h w',
                                     d=n_days, p=self.patches_per_day)
            self.mask_per_region = mask_reshaped[0]  # (patches_per_day, H, W)
            self.mask = self.mask_per_region  # Keep for backward compatibility

            lat_reshaped = rearrange(lat[:n_days * self.patches_per_day],
                                    '(d p) h w -> d p h w',
                                    d=n_days, p=self.patches_per_day)
            lon_reshaped = rearrange(lon[:n_days * self.patches_per_day],
                                    '(d p) h w -> d p h w',
                                    d=n_days, p=self.patches_per_day)
            self.lat = lat_reshaped[0]  # (patches_per_day, H, W)
            self.lon = lon_reshaped[0]  # (patches_per_day, H, W)
            self.logger.info(f"Mask format: per-timestep, extracted to per-region (patches_per_day={self.patches_per_day}, H={mask.shape[1]}, W={mask.shape[2]})")
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}. Expected (patches_per_day={self.patches_per_day}, H, W) or (N={data.shape[0]}, H, W)")

        # Apply normalization based on mode
        if self.normalization_mode == 'standardize':
            self.mean, self.std = self.compute_statistics(data, self.mask_per_region)
            data = self.normalize_data(data, self.mean, self.std)
            self.logger.info(f"Data standardized - mean shape: {self.mean.shape}, std shape: {self.std.shape}")
        elif self.normalization_mode == 'normalize':
            self.min_val, self.max_val = self.compute_statistics(data, self.mask_per_region)
            data = self.normalize_data(data, self.min_val, self.max_val)
            self.logger.info(f"Data normalized (min-max) - min shape: {self.min_val.shape}, max shape: {self.max_val.shape}")
        elif self.normalization_mode == 'none':
            self.logger.info("No normalization applied")
        else:
            raise ValueError(f"Unknown normalization_mode: {self.normalization_mode}. Must be 'standardize', 'normalize', or 'none'")

        # Create temporal sequences
        sequences, patch_indices = self.create_sequences(data, self.seq_len)
        self.logger.info(f"Created {len(sequences)} sequences with length {self.seq_len}")

        # Apply subset if needed
        if subset:
            n_subset = int(len(sequences) * sub_ratio)
            sequences = sequences[:n_subset]
            patch_indices = patch_indices[:n_subset]
            self.logger.info(f"Using subset: {n_subset} sequences")

        # Generate indices for train/valid/test split (only if not already loaded)
        if self.data_indices is None:
            n_total = len(sequences)
            indices = np.arange(n_total)

            # Split indices
            n_train = int(n_total * self.train_ratio)
            n_valid = int(n_total * self.valid_ratio)

            train_indices = indices[:n_train]
            valid_indices = indices[n_train:n_train+n_valid]
            test_indices = indices[n_train+n_valid:]

            self.data_indices = {
                'train': train_indices,
                'valid': valid_indices,
                'test': test_indices
            }

            self.logger.info(f"Generated new data split - Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}")
        else:
            self.logger.info(f"Using pre-loaded data split - Train: {len(self.data_indices['train'])}, Valid: {len(self.data_indices['valid'])}, Test: {len(self.data_indices['test'])}")

        # Split into X (input) and y (target)
        X = sequences[:, :self.input_len]   # (N, T_in, C, H, W)
        y = sequences[:, self.input_len:]   # (N, T_out, C, H, W)

        # Store patch indices for mask retrieval
        self.patch_indices = patch_indices  # (N,) - which region each sequence belongs to

        # Save normalization parameters and data info
        if self.saving_path is not None:
            self.save_data_info()

        return X, y

    def load_h5_data(self, path):
        """Load data from HDF5 file"""
        with h5py.File(path, 'r') as f:
            uovo_data = f['uovo_data'][:]  # (N, 2, H, W)
            mask = f['mask'][:]            # (147, H, W) or (N, H, W)
            lat = f['lat'][:]              # (147, H, W) or (N, H, W)
            lon = f['lon'][:]              # (147, H, W) or (N, H, W)

        # Replace NaN with 0 for computation
        uovo_data = np.nan_to_num(uovo_data, nan=0.0)

        return uovo_data, mask, lat, lon

    def load_mat_data(self, path):
        """
        Load data from MAT file (e.g., Pearl River Estuary data)

        Expected format:
            u_combined: (time, H, W) - eastward velocity
            v_combined: (time, H, W) - northward velocity
            x: (H, W) - longitude grid
            y: (H, W) - latitude grid

        Returns:
            uovo_data: (N, 2, H, W) - combined u,v data
            mask: (1, H, W) - generated from NaN values
            lat: (1, H, W) - latitude grid
            lon: (1, H, W) - longitude grid
        """

        self.logger.info(f"Loading MAT file from {path}")
        mat_data = scipy.io.loadmat(path)

        u = mat_data['u_combined']  # (time, H, W)
        v = mat_data['v_combined']  # (time, H, W)


        # Generate mask from NaN values
        mask_2d = np.isnan(u) | np.isnan(v)  # (time, H, W)
        mask_combined = np.any(mask_2d, axis=0)  # (H, W) - True if any timestep has NaN
        mask = mask_combined[np.newaxis, ...]  # (1, H, W)

        # Load coordinate grids
        if 'x' in mat_data and 'y' in mat_data:
            lon_grid = mat_data['x']  # (H, W)
            lat_grid = mat_data['y']  # (H, W)

            # Transpose if needed (MATLAB uses column-major order)
            if lon_grid.shape != u.shape[1:]:
                lon_grid = lon_grid.T
                lat_grid = lat_grid.T

            lon = lon_grid[np.newaxis, ...]  # (1, H, W)
            lat = lat_grid[np.newaxis, ...]  # (1, H, W)

            self.logger.info(f"Coordinate grids - lon range: [{lon.min():.2f}, {lon.max():.2f}], lat range: [{lat.min():.2f}, {lat.max():.2f}]")
        else:
            # Create dummy coordinate grids if not available
            H, W = u.shape[1:]
            lon = np.arange(W)[np.newaxis, np.newaxis, :].repeat(H, axis=1)  # (1, H, W)
            lat = np.arange(H)[np.newaxis, :, np.newaxis].repeat(W, axis=2)  # (1, H, W)
            self.logger.warning("Coordinate grids not found in MAT file, using dummy indices")

        uovo_data = np.stack([u, v], axis=1)  # (time, 2, H, W)
        uovo_data = np.nan_to_num(uovo_data, nan=0.0)


        return uovo_data, mask, lat, lon

    def compute_statistics(self, data, mask):
        """
        Compute statistics for normalization, ignoring masked regions

        Args:
            data: (N, C, H, W) where N = days * patches_per_day
            mask: (patches_per_day, H, W) - True for land, False for ocean

        Returns:
            Statistics computed over ocean regions only
        """

        # mask: True for land, False for ocean []
        ocean_mask = ~mask  # (patches_per_day, H, W) - True for ocean

        # Expand mask to match data dimensions
        # data: (N, C, H, W) where N = days * patches_per_day
        n_total = data.shape[0]
        n_days = n_total // self.patches_per_day

        # Repeat mask for each day: (patches_per_day, H, W) -> (days, patches_per_day, H, W)
        ocean_mask_expanded = repeat(ocean_mask, 'p h w -> d p h w', d=n_days)
        # Flatten back to match data: (days, patches_per_day, H, W) -> (N, H, W)
        ocean_mask_expanded = rearrange(ocean_mask_expanded, 'd p h w -> (d p) h w')
        # Add channel dimension: (N, H, W) -> (N, 1, H, W)
        ocean_mask_expanded = ocean_mask_expanded[:n_total, None, :, :]  # (N, 1, H, W)

        # Apply mask to data
        masked_data = data * ocean_mask_expanded

        if self.normalization_mode == 'standardize':
            # Compute mean and std for standardization
            n_ocean = np.sum(ocean_mask_expanded)
            mean = np.sum(masked_data, axis=(0, 2, 3), keepdims=True) / n_ocean
            std = np.sqrt(np.sum((masked_data - mean) ** 2 * ocean_mask_expanded, axis=(0, 2, 3), keepdims=True) / n_ocean)
            return mean.squeeze(), std.squeeze()  # (C,)

        elif self.normalization_mode == 'normalize':
            # Compute min and max for min-max normalization
            # Set masked values to inf/-inf to exclude them from min/max
            masked_data_for_minmax = np.where(ocean_mask_expanded, masked_data, np.nan)
            min_val = np.nanmin(masked_data_for_minmax, axis=(0, 2, 3), keepdims=True)
            max_val = np.nanmax(masked_data_for_minmax, axis=(0, 2, 3), keepdims=True)

            return min_val.squeeze(), max_val.squeeze()  # (C,)

        else:
            # No normalization
            return None, None
    
    def normalize_data(self, data, stat1, stat2):
        """Normalize data using different methods based on normalization_mode"""
        if self.normalization_mode == 'standardize':
            # Standardization: (data - mean) / std
            mean = stat1.reshape(1, -1, 1, 1)
            std = stat2.reshape(1, -1, 1, 1)
            return (data - mean) / (std + 1e-8)
        
        elif self.normalization_mode == 'normalize':
            # Min-max normalization: (data - min) / (max - min)
            min_val = stat1.reshape(1, -1, 1, 1)
            max_val = stat2.reshape(1, -1, 1, 1)
            return (data - min_val) / (max_val - min_val + 1e-8)
        
        else:
            # No normalization
            return data

    def denormalize_data(self, data):
        """Denormalize data back to original scale"""
        if self.normalization_mode == 'none':
            return data
        
        if self.normalization_mode == 'standardize':
            if self.mean is None or self.std is None:
                return data
            
            if isinstance(data, torch.Tensor):
                mean = torch.from_numpy(self.mean).to(data.device).reshape(1, -1, 1, 1)
                std = torch.from_numpy(self.std).to(data.device).reshape(1, -1, 1, 1)
            else:
                mean = self.mean.reshape(1, -1, 1, 1)
                std = self.std.reshape(1, -1, 1, 1)
            
            return data * (std + 1e-8) + mean
        
        elif self.normalization_mode == 'normalize':
            if self.min_val is None or self.max_val is None:
                return data
            
            if isinstance(data, torch.Tensor):
                min_val = torch.from_numpy(self.min_val).to(data.device).reshape(1, -1, 1, 1)
                max_val = torch.from_numpy(self.max_val).to(data.device).reshape(1, -1, 1, 1)
            else:
                min_val = self.min_val.reshape(1, -1, 1, 1)
                max_val = self.max_val.reshape(1, -1, 1, 1)
            
            return data * (max_val - min_val + 1e-8) + min_val
        
        else:
            raise ValueError(f"Unknown normalization_mode: {self.normalization_mode}")

    def save_data_info(self):
        """Save normalization parameters, mask, and data split indices"""
        if self.saving_path is None:
            self.logger.warning("No saving_path specified, skipping save_data_info")
            return

        data_info = {
            'normalization_mode': self.normalization_mode,
            'mean': self.mean,
            'std': self.std,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'mask': self.mask,  # For backward compatibility
            'mask_per_region': self.mask_per_region,  # (patches_per_day, H, W)
            'lat': self.lat,
            'lon': self.lon,
            'data_indices': self.data_indices,
            'patches_per_day': self.patches_per_day,
            'input_len': self.input_len,
            'output_len': self.output_len,
        }

        save_path = os.path.join(self.saving_path, 'data_info.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data_info, f)

        self.logger.info(f"Saved data info to {save_path}")

    @staticmethod
    def load_data_info(saving_path):
        """Load saved normalization parameters and data info"""
        data_info_path = os.path.join(saving_path, 'data_info.pkl')
        with open(data_info_path, 'rb') as f:
            data_info = pickle.load(f)
        return data_info

    def create_sequences(self, data, seq_len):
        """
        Create temporal sequences from data

        Data structure:
        - data: (N, C, H, W) where N = total_days * patches_per_day
        - Each day has patches_per_day patches from different ocean regions
        - We create sequences for each spatial location across time

        Example:
        - If patches_per_day=147, N=53655 (365 days)
        - Location 0: [day0_patch0, day1_patch0, ..., day364_patch0]
        - Location 1: [day0_patch1, day1_patch1, ..., day364_patch1]
        - ...

        Returns:
            sequences: (n_sequences, seq_len, C, H, W)
            patch_indices: (n_sequences,) - which spatial region each sequence belongs to
        """
        n_total = data.shape[0]
        n_days = n_total // self.patches_per_day

        self.logger.info(f"Total patches: {n_total}, Days: {n_days}, Patches per day: {self.patches_per_day}")

        # Reshape to daily structure: (Days, Patches_per_day, C, H, W)
        data = data[:n_days * self.patches_per_day]  # Trim to exact days
        data = rearrange(data, '(d p) c h w -> d p c h w', d=n_days, p=self.patches_per_day)

        # Create sequences for each spatial location
        sequences = []
        patch_indices = []
        for patch_idx in range(self.patches_per_day):
            for start_day in range(n_days - seq_len + 1):
                # Extract temporal sequence for this spatial location
                seq = data[start_day:start_day+seq_len, patch_idx]  # (T, C, H, W)
                sequences.append(seq)
                patch_indices.append(patch_idx)

        self.logger.info(f"Generated {len(sequences)} sequences = {self.patches_per_day} locations × {n_days - seq_len + 1} time windows")
        return np.array(sequences, dtype=np.float32), np.array(patch_indices, dtype=np.int32)


class OceanSequenceDataset(MyBase):
    """Dataset that includes patch indices for proper mask handling"""
    def __init__(self, X, y, patch_indices=None):
        super().__init__(X, y)
        self.patch_indices = patch_indices

    def __getitem__(self, idx):
        # X: (N, T_in, C, H, W), y: (N, T_out, C, H, W)
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.patch_indices is not None:
            patch_idx = self.patch_indices[idx]
            return x, y, patch_idx
        return x, y
