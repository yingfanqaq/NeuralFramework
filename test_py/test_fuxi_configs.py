#!/usr/bin/env python
"""
Test script to compare different OceanFuxi configurations
"""

import torch
import yaml
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import OceanFuxi

def test_configuration(config_path, config_name):
    """Test a specific Fuxi configuration"""

    print(f"\n{'='*60}")
    print(f"Testing {config_name} Configuration")
    print(f"{'='*60}")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nConfiguration:")
    print(f"  - embed_dim: {config['embed_dim']}")
    print(f"  - depth: {config['depth']}")
    print(f"  - window_size: {config['window_size']}")
    print(f"  - use_3d_path: {config.get('use_3d_path', False)}")
    print(f"  - patch_size: {config.get('patch_size', [4, 4])}")

    # Create model
    try:
        model = OceanFuxi(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - Total parameters: {total_params:,}")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return None

    # Test on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {
        'name': config_name,
        'params': total_params,
        'batch_sizes': {},
        'memory_usage': {},
        'speed': {}
    }

    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8] if device == 'cuda' else [1]

    for batch_size in batch_sizes:
        try:
            # Clear cache
            if device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            model = model.to(device)

            # Create sample data
            x = torch.randn(batch_size, 7, 2, 240, 240).to(device)

            # Warm-up run
            with torch.no_grad():
                _ = model(x)

            # Measure speed (forward pass)
            if device == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()
            with torch.no_grad():
                for _ in range(5):  # Average over 5 runs
                    output = model(x)

            if device == 'cuda':
                torch.cuda.synchronize()

            avg_time = (time.time() - start_time) / 5
            fps = batch_size / avg_time  # frames per second

            # Measure memory
            if device == 'cuda':
                memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            else:
                memory_gb = 0

            results['batch_sizes'][batch_size] = 'Success'
            results['memory_usage'][batch_size] = memory_gb
            results['speed'][batch_size] = fps

            print(f"  ✓ Batch size {batch_size}: Memory={memory_gb:.2f}GB, Speed={fps:.1f} fps")

            # Test gradient computation for batch_size=1
            if batch_size == 1:
                try:
                    model.train()
                    x_grad = torch.randn(1, 7, 2, 240, 240, requires_grad=True).to(device)
                    target = torch.randn(1, 1, 2, 240, 240).to(device)

                    output = model(x_grad)
                    loss = torch.nn.MSELoss()(output, target)
                    loss.backward()

                    print(f"    ✓ Gradient computation successful")
                except Exception as e:
                    print(f"    ✗ Gradient computation failed: {str(e)[:50]}")

                # Clear gradients
                model.zero_grad()

        except torch.cuda.OutOfMemoryError:
            results['batch_sizes'][batch_size] = 'OOM'
            print(f"  ✗ Batch size {batch_size}: Out of memory")
            break
        except Exception as e:
            results['batch_sizes'][batch_size] = 'Failed'
            print(f"  ✗ Batch size {batch_size}: {str(e)[:50]}")
            break

    return results


def main():
    """Test all Fuxi configurations"""

    print("=" * 60)
    print("OceanFuxi Model Configuration Comparison")
    print("=" * 60)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n✓ GPU cache cleared")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("\n⚠️  No GPU available, using CPU")

    # Test different configurations
    configs_to_test = [
        ('configs/models/fuxi_light_conf.yaml', 'Lightweight'),
        ('configs/models/fuxi_balanced_conf.yaml', 'Balanced'),
        ('configs/models/fuxi_conf.yaml', 'Full')
    ]

    all_results = []

    for config_path, config_name in configs_to_test:
        if os.path.exists(config_path):
            results = test_configuration(config_path, config_name)
            if results:
                all_results.append(results)
        else:
            print(f"\n⚠️  Configuration file not found: {config_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary of Results")
    print(f"{'='*60}")

    if all_results:
        # Print comparison table
        print(f"\n{'Configuration':<15} {'Parameters':<15} {'Max Batch':<12} {'Memory (GB)':<12} {'Speed (fps)':<12}")
        print("-" * 66)

        for result in all_results:
            # Find max successful batch size
            max_batch = 0
            max_memory = 0
            max_speed = 0

            for batch_size in sorted(result['batch_sizes'].keys()):
                if result['batch_sizes'][batch_size] == 'Success':
                    max_batch = batch_size
                    max_memory = result['memory_usage'].get(batch_size, 0)
                    max_speed = result['speed'].get(batch_size, 0)

            params_str = f"{result['params']/1e6:.1f}M"
            memory_str = f"{max_memory:.2f}" if max_memory > 0 else "N/A"
            speed_str = f"{max_speed:.1f}" if max_speed > 0 else "N/A"

            print(f"{result['name']:<15} {params_str:<15} {max_batch:<12} {memory_str:<12} {speed_str:<12}")

        # Recommendations
        print(f"\n{'='*60}")
        print("Recommendations")
        print(f"{'='*60}")

        print("\n📊 Configuration Guidelines:")
        print("\n1. **Lightweight** (fuxi_light_conf.yaml):")
        print("   - Best for: Quick experiments, limited GPU memory")
        print("   - Use when: You need fast iteration or have <8GB GPU memory")

        print("\n2. **Balanced** (fuxi_balanced_conf.yaml):")
        print("   - Best for: Production training with good accuracy")
        print("   - Use when: You have 8-16GB GPU memory")

        print("\n3. **Full** (fuxi_conf.yaml):")
        print("   - Best for: Maximum accuracy, final models")
        print("   - Use when: You have >16GB GPU memory and time for training")

        # Find best configuration based on available memory
        if all_results:
            lightweight = next((r for r in all_results if r['name'] == 'Lightweight'), None)
            if lightweight and lightweight['batch_sizes'].get(4) == 'Success':
                print(f"\n✅ Recommended: Start with **Lightweight** configuration")
                print(f"   - Can handle batch_size=4")
                print(f"   - Fast training speed")
                print(f"   - Good for initial experiments")

    print(f"\n{'='*60}")
    print("Testing Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Use GPU 1 if available
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    main()

    print("\n🚀 To train with Fuxi model:")
    print("   1. Edit configs/pearl_river_config.yaml")
    print("   2. Change model.name to 'OceanFuxi'")
    print("   3. Adjust batch size based on the configuration")
    print("   4. Run: python main.py --config configs/pearl_river_config.yaml")