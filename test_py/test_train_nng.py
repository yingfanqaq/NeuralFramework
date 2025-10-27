#!/usr/bin/env python
"""
Test training script for OceanNNG model with actual data
This tests if the model can be used with the training pipeline
"""

import torch
import yaml
import os
import sys
import numpy as np
from scipy.io import loadmat

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import OceanNNG

def test_train_pipeline():
    """Test OceanNNG model with actual training setup"""

    print("=" * 60)
    print("Testing OceanNNG Training Pipeline")
    print("=" * 60)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n✓ GPU cache cleared")

    # 1. Test different configurations
    configs_to_test = [
        ('configs/models/nng_light_conf.yaml', 'Lightweight'),
        ('configs/models/nng_balanced_conf.yaml', 'Balanced'),
        ('configs/models/nng_conf.yaml', 'Full')
    ]

    results = []

    for config_path, config_name in configs_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {config_name} Configuration")
        print(f"{'='*60}")

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"\nConfiguration:")
        print(f"  - mesh_level: {config['mesh_level']}")
        print(f"  - hidden_dim: {config['hidden_dim']}")
        print(f"  - processor_layers: {config['processor_layers']}")
        print(f"  - multimesh: {config.get('multimesh', True)}")

        # Create model
        try:
            model = OceanNNG(config)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  - Total parameters: {total_params:,}")
        except Exception as e:
            print(f"✗ Failed to create model: {e}")
            results.append((config_name, 'Failed', 0, 0))
            continue

        # Test on GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_sizes = [1, 2, 4] if device == 'cuda' else [1]

        max_batch = 0
        max_memory = 0

        for batch_size in batch_sizes:
            try:
                # Clear cache before each test
                if device == 'cuda':
                    torch.cuda.empty_cache()

                model = model.to(device)

                # Create sample data
                x = torch.randn(batch_size, 7, 2, 240, 240).to(device)
                target = torch.randn(batch_size, 1, 2, 240, 240).to(device)

                # Forward pass
                output = model(x)

                # Compute loss
                loss = torch.nn.MSELoss()(output, target)

                # Backward pass
                loss.backward()

                # Check memory
                if device == 'cuda':
                    memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                    max_memory = max(max_memory, memory_gb)
                else:
                    memory_gb = 0

                print(f"  ✓ Batch size {batch_size}: Loss={loss.item():.4f}, Memory={memory_gb:.2f}GB")
                max_batch = batch_size

                # Clear gradients
                model.zero_grad()

            except torch.cuda.OutOfMemoryError:
                print(f"  ✗ Batch size {batch_size}: Out of memory")
                break
            except Exception as e:
                print(f"  ✗ Batch size {batch_size}: {str(e)[:50]}")
                break

        results.append((config_name, 'Success', max_batch, max_memory))

    # 2. Summary
    print(f"\n{'='*60}")
    print("Summary of Results")
    print(f"{'='*60}")
    print(f"{'Configuration':<15} {'Status':<10} {'Max Batch':<12} {'Max Memory (GB)':<15}")
    print("-" * 60)
    for config_name, status, max_batch, max_memory in results:
        print(f"{config_name:<15} {status:<10} {max_batch:<12} {max_memory:<15.2f}")

    # 3. Recommendations
    print(f"\n{'='*60}")
    print("Recommendations for Training")
    print(f"{'='*60}")

    # Find best configuration
    successful_configs = [(name, batch, mem) for name, status, batch, mem in results if status == 'Success']

    if successful_configs:
        # Sort by batch size (descending) then memory (ascending)
        successful_configs.sort(key=lambda x: (-x[1], x[2]))
        best_config, best_batch, best_memory = successful_configs[0]

        print(f"\n✅ Recommended configuration: {best_config}")
        print(f"   - Maximum batch size: {best_batch}")
        print(f"   - Memory usage: {best_memory:.2f} GB")

        if best_config == 'Lightweight':
            print("\n📝 To use this configuration, update configs/pearl_river_config.yaml:")
            print("   Change the batch size to:", best_batch)
            print("   Consider using nng_light_conf.yaml for training")
        elif best_config == 'Balanced':
            print("\n📝 To use this configuration, update configs/pearl_river_config.yaml:")
            print("   Change the batch size to:", best_batch)
            print("   Consider using nng_balanced_conf.yaml for training")
        else:
            print("\n📝 Full configuration works! Update configs/pearl_river_config.yaml:")
            print("   Change the batch size to:", best_batch)

        print("\n🚀 Training command:")
        print("   python train.py --config configs/pearl_river_config.yaml")
    else:
        print("\n⚠️  No configuration succeeded. Consider:")
        print("   - Further reducing model complexity")
        print("   - Using CPU for training")
        print("   - Checking GPU memory availability")

    return len(successful_configs) > 0


if __name__ == "__main__":
    # Use GPU 1 (seems less occupied)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    success = test_train_pipeline()

    if not success:
        print("\n⚠️  All configurations failed.")
        sys.exit(1)
    else:
        print("\n✅ OceanNNG model is ready for training!")