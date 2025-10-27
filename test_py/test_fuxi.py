#!/usr/bin/env python
"""
Test script for OceanFuxi model
"""

import torch
import yaml
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import OceanFuxi

def test_fuxi_model():
    """Test OceanFuxi model initialization and forward pass"""

    print("=" * 60)
    print("Testing OceanFuxi Model")
    print("=" * 60)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("\n✓ GPU cache cleared")

    # 1. Load configuration
    print("\n1. Loading configuration from fuxi_conf.yaml...")
    config_path = 'configs/models/fuxi_conf.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"   Configuration loaded:")
    print(f"   - embed_dim: {config['embed_dim']}")
    print(f"   - depth: {config['depth']}")
    print(f"   - window_size: {config['window_size']}")
    print(f"   - use_3d_path: {config['use_3d_path']}")
    print(f"   - pseudo_depth: {config['pseudo_depth']}")

    # 2. Create model
    print("\n2. Creating OceanFuxi model...")
    try:
        model = OceanFuxi(config)
        print(f"   ✓ Model created successfully")
        print(f"   Model type: {type(model).__name__}")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Check model structure
    print("\n3. Model structure:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # 4. Test forward pass with different devices
    print("\n4. Testing forward pass...")

    # Test on CPU first
    print("\n   Testing on CPU:")
    device = 'cpu'
    model = model.to(device)

    try:
        x = torch.randn(1, 7, 2, 240, 240).to(device)
        print(f"     Input shape: {x.shape}")

        with torch.no_grad():
            output = model(x)

        print(f"     Output shape: {output.shape}")
        print(f"     ✓ CPU forward pass successful")
    except Exception as e:
        print(f"     ✗ CPU forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test on GPU if available
    if torch.cuda.is_available():
        print("\n   Testing on GPU:")
        device = 'cuda'

        # Clear cache before GPU test
        torch.cuda.empty_cache()

        try:
            model = model.to(device)
            print("     ✓ Model moved to GPU")
        except Exception as e:
            print(f"     ✗ Failed to move model to GPU: {e}")
            return False

        # Test with batch size 1
        try:
            x = torch.randn(1, 7, 2, 240, 240).to(device)
            print(f"     Input shape: {x.shape}")

            with torch.no_grad():
                output = model(x)

            print(f"     Output shape: {output.shape}")
            print(f"     ✓ GPU forward pass successful (batch_size=1)")

            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"     Memory allocated: {memory_allocated:.2f} GB")
            print(f"     Memory reserved: {memory_reserved:.2f} GB")

        except torch.cuda.OutOfMemoryError:
            print(f"     ✗ GPU forward pass failed: Out of memory")
            return False
        except Exception as e:
            print(f"     ✗ GPU forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Test with batch size 2
        print("\n   Testing with batch_size=2:")
        try:
            torch.cuda.empty_cache()
            x = torch.randn(2, 7, 2, 240, 240).to(device)
            print(f"     Input shape: {x.shape}")

            with torch.no_grad():
                output = model(x)

            print(f"     Output shape: {output.shape}")
            print(f"     ✓ Forward pass successful (batch_size=2)")

            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"     Memory allocated: {memory_allocated:.2f} GB")

        except torch.cuda.OutOfMemoryError:
            print(f"     ⚠️ Batch size 2 causes OOM (consider using smaller batch size)")
        except Exception as e:
            print(f"     ✗ Forward pass failed: {e}")

    # 5. Test gradient computation
    print("\n5. Testing gradient computation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        torch.cuda.empty_cache()

    try:
        model.train()
        x = torch.randn(1, 7, 2, 240, 240, requires_grad=True).to(device)
        target = torch.randn(1, 1, 2, 240, 240).to(device)

        output = model(x)
        loss = torch.nn.MSELoss()(output, target)
        loss.backward()

        # Check if gradients are computed
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        if has_grad:
            print("   ✓ Gradients computed successfully")
            print(f"   Loss value: {loss.item():.4f}")
        else:
            print("   ✗ No gradients computed")

    except torch.cuda.OutOfMemoryError:
        print("   ⚠️ Gradient computation causes OOM (consider using smaller model or batch size)")
    except Exception as e:
        print(f"   ✗ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()

    # 6. Memory usage summary
    if torch.cuda.is_available():
        print("\n6. Memory usage summary:")
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   Peak GPU memory: {peak_memory:.2f} GB")

        if peak_memory > 20:
            print("   ⚠️ High memory usage detected. Consider:")
            print("      - Using lightweight configuration")
            print("      - Reducing batch size")
            print("      - Using gradient accumulation")
            print("      - Enabling mixed precision training")

    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    # Use GPU 1 if available (GPU 0 seems occupied)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    success = test_fuxi_model()

    if not success:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ OceanFuxi model is working correctly!")
        print("\nYou can now train the model using:")
        print("  python main.py --config configs/pearl_river_config.yaml")
        print("  (Remember to change model.name to 'OceanFuxi' in the config)")