#!/usr/bin/env python
"""
Test script for OceanNNG model
"""

import torch
import yaml
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import OceanNNG

def test_nng_model():
    """Test OceanNNG model initialization and forward pass"""

    print("=" * 60)
    print("Testing OceanNNG Model")
    print("=" * 60)

    # 1. Load configuration
    print("\n1. Loading configuration from nng_conf.yaml...")
    config_path = 'configs/models/nng_conf.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"   Configuration loaded: {config}")

    # 2. Create model
    print("\n2. Creating OceanNNG model...")
    try:
        model = OceanNNG(config)
        print(f"   ✓ Model created successfully")
        print(f"   Model type: {type(model).__name__}")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False

    # 3. Check model structure
    print("\n3. Model structure:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # 4. Test forward pass with different batch sizes
    print("\n4. Testing forward pass...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")

    # Move model to device
    try:
        model = model.to(device)
        print("   ✓ Model moved to device")
    except Exception as e:
        print(f"   ✗ Failed to move model to device: {e}")
        device = 'cpu'
        model = model.to(device)

    # Test with batch size 1
    print("\n   Testing with batch_size=1:")
    try:
        x = torch.randn(1, 7, 2, 240, 240).to(device)
        print(f"     Input shape: {x.shape}")

        with torch.no_grad():
            output = model(x)

        print(f"     Output shape: {output.shape}")
        print(f"     ✓ Forward pass successful (batch_size=1)")
    except Exception as e:
        print(f"     ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with batch size 2
    print("\n   Testing with batch_size=2:")
    try:
        x = torch.randn(2, 7, 2, 240, 240).to(device)
        print(f"     Input shape: {x.shape}")

        with torch.no_grad():
            output = model(x)

        print(f"     Output shape: {output.shape}")
        print(f"     ✓ Forward pass successful (batch_size=2)")
    except Exception as e:
        print(f"     ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. Test gradient computation
    print("\n5. Testing gradient computation...")
    try:
        model.train()
        x = torch.randn(1, 7, 2, 240, 240).to(device)
        target = torch.randn(1, 1, 2, 240, 240).to(device)

        output = model(x)
        loss = torch.nn.MSELoss()(output, target)
        loss.backward()

        # Check if gradients are computed
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        if has_grad:
            print("   ✓ Gradients computed successfully")
        else:
            print("   ✗ No gradients computed")
    except Exception as e:
        print(f"   ✗ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. Memory usage estimation
    print("\n6. Memory usage (approximate):")
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        x = torch.randn(1, 7, 2, 240, 240).to(device)
        with torch.no_grad():
            _ = model(x)

        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   Peak GPU memory: {memory_allocated:.2f} GB")
    else:
        print("   Running on CPU, GPU memory stats not available")

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    # Set environment for testing
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU if available

    success = test_nng_model()

    if not success:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ OceanNNG model is working correctly!")
        print("\nYou can now train the model using:")
        print("  python train.py --config configs/pearl_river_config.yaml")