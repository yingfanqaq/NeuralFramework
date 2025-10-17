import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from einops import repeat

from .base import BaseTrainer
from utils import LossRecord
from utils.metrics import METRIC_DICT, VALID_METRICS


class OceanTrainer(BaseTrainer):
    """Trainer for ocean velocity prediction task"""

    def __init__(self, args):
        super().__init__(args)

        self.dataset = None # Store dataset for visualization

        for dataset_name, dataset_obj in [('train', self.train_loader.dataset),
                                            ('valid', self.valid_loader.dataset),
                                            ('test', self.test_loader.dataset)]:
            if hasattr(dataset_obj, 'dataset'):
                parent = dataset_obj.dataset
                if hasattr(parent, 'mask'):
                    self.mask = parent.mask
                    self.lat = parent.lat
                    self.lon = parent.lon
                    self.dataset = parent
                    break

        # Load model if model_path is provided
        self.start_epoch = 0
        model_path = args.get('model_path', None)
        if model_path is not None and os.path.exists(model_path):
            self._load_model(model_path)
            self.logger.info(f"Loaded model from {model_path}")

    def _load_model(self, model_path):
        """Load model from checkpoint file

        Args:
            model_path: Path to model file (.pth)
        """
        if isinstance(self.device, int):
          map_location = f'cuda:{self.device}'
        elif self.device == 'cpu':
            map_location = 'cpu'
        else:
            map_location = self.device

        breakpoint()
        checkpoint = torch.load(model_path, map_location=map_location)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Format: {'epoch': ..., 'model_state_dict': ..., 'optimizer_state_dict': ..., ...}
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # Load optimizer state if available and in train mode
                if 'optimizer_state_dict' in checkpoint and self.args.get('mode', 'train') == 'train':
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.logger.info("Loaded optimizer state for continued training")

                # Set start epoch if available
                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch'] + 1
                    self.logger.info(f"Resuming from epoch {self.start_epoch}")
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

    def _apply_mask_and_compute_loss(self, y_pred, y, patch_idx):
        """Apply per-region mask and compute loss (only on ocean pixels)

        Args:
            y_pred: (B, T_out, C, H, W) - predictions
            y: (B, T_out, C, H, W) - targets
            patch_idx: (B,) - region indices

        Returns:
            loss: scalar - MSE loss computed only on ocean pixels
            mae: scalar - MAE computed only on ocean pixels (or None)
        """

        if hasattr(self.dataset, 'mask_per_region') and self.dataset.mask_per_region is not None:
            # Get masks for this batch: (B, H, W) - True for land, False for ocean
            batch_masks = self.dataset.mask_per_region[patch_idx]
            # Convert to ocean mask: True for ocean, False for land
            ocean_mask = torch.from_numpy(~batch_masks).to(self.device).float()  # (B, H, W)
            # Expand to match prediction shape: (B, H, W) -> (B, T_out, C, H, W)
            ocean_mask = repeat(ocean_mask, 'b h w -> b t c h w',
                              t=y_pred.shape[1], c=y_pred.shape[2])

            # MSE loss: only divide by ocean pixels
            squared_error = (y_pred - y) ** 2 * ocean_mask
            mse_loss = squared_error.sum() / (ocean_mask.sum() + 1e-8)

            # MAE: optional
            abs_error = torch.abs(y_pred - y) * ocean_mask
            mae = abs_error.sum() / (ocean_mask.sum() + 1e-8)

            return mse_loss, mae, ocean_mask
        else:
            # No mask - use all pixels
            mse_loss = torch.nn.functional.mse_loss(y_pred, y)
            mae = torch.nn.functional.l1_loss(y_pred, y)
            return mse_loss, mae, None

    def train(self, epoch):
        """Train for one epoch with proper per-region mask handling"""
        loss_record = LossRecord(["train_loss"])
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch data
            if len(batch_data) == 3:
                x, y, patch_idx = batch_data
                patch_idx = patch_idx.numpy()  # (B,)
            else:
                raise ValueError("Batch data must include patch_idx. Check dataset implementation.")

            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(x)  # (B, T_out, C, H, W)
            loss, _, _ = self._apply_mask_and_compute_loss(y_pred, y, patch_idx)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_record.update({"train_loss": loss.item()}, n=1)
            pbar.set_postfix({'loss': f'{loss.item():.2e}', 'avg': f'{loss_record.loss_dict["train_loss"].avg:.2e}'})

        if self.scheduler is not None:
            self.scheduler.step()

        return loss_record

    def predict(self, x):
        """Make predictions and return both normalized and denormalized results"""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y_pred_norm = self.model(x)

            if self.dataset is not None and hasattr(self.dataset, 'denormalize_data'):
                y_pred_real = self.dataset.denormalize_data(y_pred_norm)
            else:
                y_pred_real = y_pred_norm

        return y_pred_norm, y_pred_real

    def evaluate(self, split="valid", return_predictions=False, save_predictions=False):
        """Evaluate on validation or test set using real (denormalized) data with proper mask handling
        Args:
            split: 'valid' or 'test'
        Returns:
            loss_record: LossRecord object
            predictions_data: dict with all predictions (only if return_predictions=True)
        """
        metric_names = [f"{split}_loss"]  # Loss for progress tracking

        # All metrics computed from global data
        for metric in VALID_METRICS:
            metric_names.append(f"{split}_{metric}")

        loss_record = LossRecord(metric_names)
        self.model.eval()

        eval_loader = self.valid_loader if split == "valid" else self.test_loader

        # Store all predictions and targets for metric calculation -> (N*c*h*w, )
        all_predictions = []
        all_targets = []

        # Store full batch data if needed for saving -> Orginal shape
        if return_predictions:
            all_inputs = []
            all_predictions_full = []
            all_targets_full = []
            all_patch_indices = []

        with torch.no_grad():
            for batch_data in tqdm(eval_loader, desc=f"Eval {split}"):
                if len(batch_data) == 3:
                    x, y, patch_idx = batch_data
                    patch_idx = patch_idx.numpy()  # (B,)
                else:
                    raise ValueError("Batch data must include patch_idx. Check dataset implementation.")

                x = x.to(self.device)
                y = y.to(self.device)

                # Get both normalized and real predictions
                _, y_pred_real = self.predict(x)

                # Denormalize targets to get real values
                if self.dataset is not None and hasattr(self.dataset, 'denormalize_data'):
                    y_real = self.dataset.denormalize_data(y)
                else:
                    y_real = y

                # Store full batch data if needed
                if return_predictions:
                    all_inputs.append(x.cpu().numpy())
                    all_predictions_full.append(y_pred_real.cpu().numpy())
                    all_targets_full.append(y_real.cpu().numpy())
                    all_patch_indices.append(patch_idx)

                # Use the unified mask application method
                loss, mae, ocean_mask = self._apply_mask_and_compute_loss(y_pred_real, y_real, patch_idx)

                # Extract ocean pixels for metrics calculation
                if ocean_mask is not None:
                    masked_pred = y_pred_real[ocean_mask.bool()].cpu().numpy()
                    masked_target = y_real[ocean_mask.bool()].cpu().numpy()
                else:
                    masked_pred = y_pred_real.cpu().numpy().reshape(-1)
                    masked_target = y_real.cpu().numpy().reshape(-1)

                # Accumulate for global metrics
                all_predictions.extend(masked_pred)
                all_targets.extend(masked_target)

                # Only track loss for progress monitoring (not used in final metrics)
                loss_record.update({
                    f"{split}_loss": loss.item(),
                }, n=1)

        # Calculate all metrics using global accumulated data
        if len(all_predictions) > 0:
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)

            # Compute all metrics from global data
            for metric_name in VALID_METRICS:
                try:
                    metric_func = METRIC_DICT[metric_name]
                    metric_value = metric_func(all_targets, all_predictions)
                    loss_record.update({f"{split}_{metric_name}": metric_value}, n=1)
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {metric_name}: {e}")
                    loss_record.update({f"{split}_{metric_name}": float('nan')}, n=1)

        self.logger.info(f"[yingfanqaq]{split.capitalize()} |{loss_record}")

        if return_predictions:
            predictions_data = {
                'inputs': np.concatenate(all_inputs, axis=0),
                'predictions': np.concatenate(all_predictions_full, axis=0),
                'targets': np.concatenate(all_targets_full, axis=0),
                'patch_indices': np.concatenate(all_patch_indices, axis=0)
            }
            if save_predictions:
                self._save_predictions(predictions_data, split)

            return loss_record, predictions_data
        else:
            if save_predictions:
                assert "you must set the save_predictions=True"

        return loss_record

    def process(self):
        """Main training loop"""
        self.logger.info("=" * 80)
        self.logger.info("Starting Training")
        self.logger.info(f"Epochs: {self.epochs} | Eval Freq: {self.eval_freq} | Patience: {self.patience}")
        if self.start_epoch > 0:
            self.logger.info(f"Resuming from epoch {self.start_epoch}")
        self.logger.info("=" * 80)

        best_epoch = 0
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(self.start_epoch, self.epochs):
            # Train
            train_loss_record = self.train(epoch)
            self.logger.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss_record.loss_dict['train_loss'].avg:.2e} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if self.wandb:
                import wandb
                wandb.log({'epoch': epoch, **train_loss_record.to_dict()})

            if self.saving_ckpt and (epoch + 1) % self.ckpt_freq == 0:
                ckpt_path = os.path.join(self.saving_path, f"checkpoint_{epoch}.pth")
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss_record.to_dict(),
                    'valid_metrics': None,
                }, ckpt_path)
                self.logger.info(f"Checkpoint saved: epoch {epoch}")

            if (epoch + 1) % self.eval_freq == 0:
                valid_loss_record = self.evaluate(split="valid")

                if self.wandb:
                    import wandb
                    wandb.log({'epoch': epoch, **valid_loss_record.to_dict()})

                # Check for improvement
                val_loss = valid_loss_record.to_dict()['valid_loss']
                if val_loss < best_val_loss:
                    improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 0
                    self.logger.info(f"New best model! Improvement: {improvement:.2f}%")
                    best_val_loss = val_loss
                    best_epoch = epoch
                    counter = 0

                    if self.saving_best:
                        best_model_path = os.path.join(self.saving_path, "best_model.pth")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'train_loss': train_loss_record.to_dict(),
                            'valid_metrics': valid_loss_record.to_dict(),
                        }, best_model_path)
                else:
                    counter += 1
                    if self.patience != -1 and counter >= self.patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break

        # Final evaluation
        self.logger.info("=" * 80)
        self.logger.info("Training Completed!")
        self.logger.info("=" * 80)

        if self.saving_best and os.path.exists(os.path.join(self.saving_path, "best_model.pth")):
            checkpoint = torch.load(os.path.join(self.saving_path, "best_model.pth"))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best model from epoch {best_epoch+1}")


        # Save final test metrics
        if self.saving_best:
            metrics_path = os.path.join(self.saving_path, "final_metrics.npz")
            np.savez(metrics_path,
                     valid_metrics=self.evaluate(split="valid", return_predictions=True, save_predictions=True)[0].to_dict(),
                     test_metrics=self.evaluate(split="test", return_predictions=True, save_predictions=True)[0].to_dict())
            self.logger.info(f"Saved final metrics to {metrics_path}")

        if self.wandb:
            import wandb
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary.update(test_loss_record.to_dict())

        self.logger.info("=" * 80)

    def test(self):
        """Test mode - save predictions for visualization"""
        self.logger.info("=" * 80)
        self.logger.info("Running Test Mode")
        self.logger.info("=" * 80)

        # Model is already loaded in __init__ via model_path
        # Evaluate and get predictions in one pass
        test_loss_record, predictions_data = self.evaluate(split="test", return_predictions=True, save_predictions=True)

        # Save test metrics
        test_metrics_path = os.path.join(self.saving_path, "test_metrics.npz")
        np.savez(test_metrics_path, test_metrics=test_loss_record.to_dict())
        self.logger.info(f"Saved test metrics to {test_metrics_path}")

        self.logger.info("=" * 80)
        self.logger.info("Test completed. Use visualize_ocean.py for visualization.")
        self.logger.info("=" * 80)

    def _save_predictions(self, predictions_data, split):
        """Save predictions and ground truth for visualization

        Args:
            predictions_data: dict with keys 'inputs', 'predictions', 'targets', 'patch_indices'
        """
        self.logger.info("Saving predictions for visualization...")

        # Create predictions directory
        pred_dir = os.path.join(self.saving_path, f"{split}_predictions")
        os.makedirs(pred_dir, exist_ok=True)

        # Save metadata
        metadata = {
            'mask': self.mask if hasattr(self, 'mask') else None,
            'mask_per_region': self.dataset.mask_per_region if hasattr(self.dataset, 'mask_per_region') else None,
            'lat': self.lat if hasattr(self, 'lat') else None,
            'lon': self.lon if hasattr(self, 'lon') else None,
            'patches_per_day': self.dataset.patches_per_day if hasattr(self.dataset, 'patches_per_day') else None,
            'num_samples': len(predictions_data['predictions']),
            'normalization_mode': getattr(self.dataset, 'normalization_mode', None) if hasattr(self.dataset, 'normalization_mode') else None
        }

        # Save all data
        np.savez(os.path.join(pred_dir, "all_predictions.npz"), **predictions_data)
        np.savez(os.path.join(pred_dir, "metadata.npz"), **metadata)

        self.logger.info(f"Saved {len(predictions_data['predictions'])} samples to {pred_dir}")
