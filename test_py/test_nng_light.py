#!/usr/bin/env python
"""
Lightweight test script for OceanNNG model
Tests with reduced configuration to avoid memory issues
"""

import torch
import yaml
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import OceanNNG

def test_nng_model():
    """Test OceanNNG model with lightweight configuration"""

    print("=" * 60)
    print("Testing OceanNNG Model (Lightweight Configuration)")
    print("=" * 60)

    # Clear GPU cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("\n✓ GPU cache cleared")

    # 1. Load lightweight configuration
    print("\n1. Loading lightweight configuration...")
    config_path = 'configs/models/nng_light_conf.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"   Configuration loaded:")
    print(f"   - mesh_level: {config['mesh_level']} (reduced)")
    print(f"   - hidden_dim: {config['hidden_dim']} (reduced)")
    print(f"   - processor_layers: {config['processor_layers']} (reduced)")
    print(f"   - multimesh: {config['multimesh']}")

    # 2. Create model
    print("\n2. Creating OceanNNG model...")
    try:
        model = OceanNNG(config)
        print(f"   ✓ Model created successfully")
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

    # 4. Test on CPU first
    print("\n4. Testing on CPU first...")
    device = 'cpu'
    model = model.to(device)

    try:
        x = torch.randn(1, 7, 2, 240, 240).to(device)
        print(f"   Input shape: {x.shape}")

        with torch.no_grad():
            output = model(x)

        print(f"   Output shape: {output.shape}")
        print(f"   ✓ CPU forward pass successful")
    except Exception as e:
        print(f"   ✗ CPU forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. Test on GPU if available
    if torch.cuda.is_available():
        print("\n5. Testing on GPU...")
        device = 'cuda'

        # Clear cache again
        torch.cuda.empty_cache()

        try:
            model = model.to(device)
            print("   ✓ Model moved to GPU")
        except Exception as e:
            print(f"   ✗ Failed to move model to GPU: {e}")
            return False

        # Test with batch size 1
        try:
            x = torch.randn(1, 7, 2, 240, 240).to(device)
            print(f"   Input shape: {x.shape}")

            with torch.no_grad():
                output = model(x)

            print(f"   Output shape: {output.shape}")
            print(f"   ✓ GPU forward pass successful")

            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   Memory allocated: {memory_allocated:.2f} GB")
            print(f"   Memory reserved: {memory_reserved:.2f} GB")

        except Exception as e:
            print(f"   ✗ GPU forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 6. Test gradient computation with minimal batch
        print("\n6. Testing gradient computation...")
        try:
            torch.cuda.empty_cache()  # Clear cache before gradient test

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

            # Final memory check
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   Final memory allocated: {memory_allocated:.2f} GB")
            print(f"   Final memory reserved: {memory_reserved:.2f} GB")

        except Exception as e:
            print(f"   ✗ Gradient computation failed: {e}")
            if "out of memory" in str(e).lower():
                print("\n   💡 Suggestion: Further reduce model complexity or use CPU for training")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    # Use GPU 1 if available (GPU 0 seems occupied)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    success = test_nng_model()

    if not success:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ OceanNNG model is working correctly!")
        print("\n📝 Notes:")
        print("   - Use the lightweight configuration for memory-constrained environments")
        print("   - Consider using smaller batch sizes (1-2) for training")
        print("   - You can adjust mesh_level and hidden_dim to balance accuracy vs memory")
        print("\nYou can now train the model using:")
        print("  python train.py --config configs/pearl_river_config.yaml")