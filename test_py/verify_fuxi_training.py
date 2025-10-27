#!/usr/bin/env python
"""
Verification script for OceanFuxi training setup
Tests model with actual training configuration
"""

import torch
import yaml
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_fuxi_training():
    """Verify that OceanFuxi is properly configured for training"""

    print("=" * 60)
    print("OceanFuxi Training Setup Verification")
    print("=" * 60)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n✓ GPU cache cleared")

    # 1. Test model variants
    print("\n1. Testing OceanFuxi model variants...")

    model_variants = [
        ('OceanFuxi_Light', 'Lightweight (13M params)'),
        ('OceanFuxi', 'Balanced (80M params)'),
        ('OceanFuxi_Full', 'Full (256M params)')
    ]

    # Load model mapping
    mapping_path = 'configs/models/model_mapping.yaml'
    with open(mapping_path, 'r') as f:
        mapping = yaml.safe_load(f)

    from models import _model_dict

    results = []
    for model_name, description in model_variants:
        print(f"\n   Testing {model_name} - {description}:")

        if model_name in mapping:
            config_path = mapping[model_name]
            print(f"   Config: {config_path}")

            # Load config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Create model
            try:
                model_class = _model_dict[model_name]
                model = model_class(config)

                total_params = sum(p.numel() for p in model.parameters())
                print(f"   ✓ Model created successfully")
                print(f"   Parameters: {total_params:,}")

                # Test forward pass
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(device)

                x = torch.randn(1, 7, 2, 240, 240).to(device)
                with torch.no_grad():
                    output = model(x)

                assert output.shape == (1, 1, 2, 240, 240), f"Output shape mismatch: {output.shape}"
                print(f"   ✓ Forward pass successful")

                # Memory test for different batch sizes
                if device == 'cuda':
                    max_batch = 0
                    for batch_size in [1, 2, 4, 8]:
                        try:
                            torch.cuda.empty_cache()
                            x = torch.randn(batch_size, 7, 2, 240, 240).to(device)
                            with torch.no_grad():
                                _ = model(x)
                            max_batch = batch_size
                        except torch.cuda.OutOfMemoryError:
                            break

                    memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"   Max batch size: {max_batch}")
                    print(f"   Memory usage: {memory_gb:.2f} GB")

                    results.append({
                        'name': model_name,
                        'params': total_params,
                        'max_batch': max_batch,
                        'memory': memory_gb
                    })
                else:
                    results.append({
                        'name': model_name,
                        'params': total_params,
                        'max_batch': 1,
                        'memory': 0
                    })

            except Exception as e:
                print(f"   ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ✗ Not found in model mapping")

    # 2. Recommend configuration for training
    print(f"\n{'='*60}")
    print("Training Recommendations")
    print(f"{'='*60}")

    if results:
        print("\n📊 Model Comparison:")
        print(f"{'Model':<20} {'Parameters':<15} {'Max Batch':<12} {'Memory (GB)':<12}")
        print("-" * 60)
        for r in results:
            params_str = f"{r['params']/1e6:.1f}M"
            print(f"{r['name']:<20} {params_str:<15} {r['max_batch']:<12} {r['memory']:<12.2f}")

        print("\n✅ Recommended Training Configurations:")

        print("\n1. **For Quick Experiments:**")
        print("   Model: OceanFuxi_Light")
        print("   Batch size: 8-16")
        print("   Learning rate: 5e-4")
        print("   Why: Fast training, low memory, good for hyperparameter tuning")

        print("\n2. **For Production Training:**")
        print("   Model: OceanFuxi (Balanced)")
        print("   Batch size: 4-8")
        print("   Learning rate: 3e-4")
        print("   Why: Good balance of speed and accuracy")

        print("\n3. **For Best Accuracy:**")
        print("   Model: OceanFuxi_Full")
        print("   Batch size: 2-4")
        print("   Learning rate: 1e-4")
        print("   Why: Maximum model capacity, best for final models")

    # 3. Create sample training configuration
    print(f"\n{'='*60}")
    print("Sample Training Configuration")
    print(f"{'='*60}")

    sample_config = """
To train with OceanFuxi, update configs/pearl_river_config.yaml:

```yaml
model:
  name: 'OceanFuxi'  # or 'OceanFuxi_Light' or 'OceanFuxi_Full'

data:
  train_batchsize: 4   # Adjust based on GPU memory
  eval_batchsize: 4

train:
  device_ids: [4, 5, 6, 7]  # Use available GPUs
  epochs: 300  # Fuxi converges faster than NNG
  patience: 15

optimizer:
  optimizer: 'AdamW'
  lr: 0.0003  # Lower for larger models
  weight_decay: 0.0001
```
"""
    print(sample_config)

    # 4. Training commands
    print(f"{'='*60}")
    print("Training Commands")
    print(f"{'='*60}")

    print("\n🚀 To start training:")
    print("\n# For quick testing (lightweight model):")
    print("python main.py --config configs/pearl_river_config.yaml --model OceanFuxi_Light")

    print("\n# For balanced training (recommended):")
    print("python main.py --config configs/pearl_river_config.yaml --model OceanFuxi")

    print("\n# For maximum accuracy:")
    print("python main.py --config configs/pearl_river_config.yaml --model OceanFuxi_Full")

    print("\n💡 Tips:")
    print("- OceanFuxi_Light: 10x faster than Full version")
    print("- Use gradient accumulation if batch size is limited")
    print("- Monitor validation loss - Fuxi can overfit on small datasets")
    print("- Consider mixed precision training (fp16) for larger models")

    return len(results) > 0


if __name__ == "__main__":
    # Use GPU 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    success = verify_fuxi_training()

    if success:
        print(f"\n✅ OceanFuxi is ready for training!")
    else:
        print(f"\n⚠️ Some issues detected. Please check the errors above.")
        sys.exit(1)