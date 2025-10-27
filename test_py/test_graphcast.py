"""
Test script for GraphCast model
Validates the implementation with your project's data format
"""

import torch
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.graphcast import GraphCast, GraphCastAutoregressive


def test_graphcast_model():
    """Test GraphCast model with various configurations"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    print("=" * 60)

    # Test configurations
    configs = {
        "Light": {
            'input_len': 7,
            'output_len': 1,
            'in_channels': 2,
            'input_res': [240, 240],
            'hidden_dim': 64,
            'mesh_level': 3,
            'processor_layers': 4,
            'mlp_layers': 1,
            'multimesh': False,
            'aggregation': 'mean',
            'add_3d_dim': False,
            'temporal_encoding': 'concat'
        },
        "Standard": {
            'input_len': 7,
            'output_len': 1,
            'in_channels': 2,
            'input_res': [240, 240],
            'hidden_dim': 128,
            'mesh_level': 4,
            'processor_layers': 8,
            'mlp_layers': 1,
            'multimesh': True,
            'aggregation': 'sum',
            'add_3d_dim': True,
            'temporal_encoding': 'concat'
        }
    }

    # Test each configuration
    for config_name, config in configs.items():
        print(f"\nTesting {config_name} Configuration")
        print("-" * 40)

        try:
            # Create model
            model = GraphCast(config).to(device)

            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {param_count:,}")

            # Create test input
            batch_size = 2
            x = torch.randn(
                batch_size,
                config['input_len'],
                config['in_channels'],
                config['input_res'][0],
                config['input_res'][1]
            ).to(device)

            print(f"Input shape: {x.shape}")

            # Forward pass
            with torch.no_grad():
                y = model(x)

            expected_shape = (
                batch_size,
                config['output_len'],
                config['in_channels'],
                config['input_res'][0],
                config['input_res'][1]
            )

            print(f"Output shape: {y.shape}")
            print(f"Expected shape: {expected_shape}")

            # Validate output
            assert y.shape == expected_shape, f"Shape mismatch! Got {y.shape}, expected {expected_shape}"
            assert not torch.isnan(y).any(), "Output contains NaN values!"
            assert not torch.isinf(y).any(), "Output contains Inf values!"

            # Check memory usage
            if device == "cuda":
                memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()
                print(f"Peak GPU memory: {memory_mb:.1f} MB")

            print(f"✓ {config_name} configuration test passed!")

        except Exception as e:
            print(f"✗ {config_name} configuration test failed!")
            print(f"Error: {str(e)}")
            raise

    print("\n" + "=" * 60)
    print("Testing Autoregressive Mode")
    print("=" * 60)

    # Test autoregressive model
    auto_config = {
        'input_len': 7,
        'output_len': 1,
        'in_channels': 2,
        'input_res': [240, 240],
        'hidden_dim': 64,
        'mesh_level': 3,
        'processor_layers': 4,
        'mlp_layers': 1,
        'multimesh': True,
        'aggregation': 'sum',
        'add_3d_dim': False,
        'temporal_encoding': 'concat',
        'rollout_steps': 5
    }

    try:
        model_auto = GraphCastAutoregressive(auto_config).to(device)

        # Test input
        x = torch.randn(1, 7, 2, 240, 240).to(device)

        # Test different rollout steps
        for rollout in [1, 3, 5]:
            with torch.no_grad():
                y = model_auto(x, rollout_steps=rollout)

            expected_shape = (1, rollout, 2, 240, 240)
            print(f"Rollout {rollout} steps: {y.shape} (expected {expected_shape})")

            assert y.shape == expected_shape, f"Shape mismatch for rollout={rollout}"
            assert not torch.isnan(y).any(), f"NaN in output for rollout={rollout}"

        print("✓ Autoregressive mode test passed!")

    except Exception as e:
        print(f"✗ Autoregressive mode test failed!")
        print(f"Error: {str(e)}")
        raise

    print("\n" + "=" * 60)
    print("Testing Multi-frame Output")
    print("=" * 60)

    # Test multi-frame output configuration
    multi_config = {
        'input_len': 7,
        'output_len': 3,  # Multiple output frames
        'in_channels': 2,
        'input_res': [240, 240],
        'hidden_dim': 64,
        'mesh_level': 3,
        'processor_layers': 4,
        'mlp_layers': 1,
        'multimesh': False,
        'aggregation': 'sum',
        'add_3d_dim': True,
        'temporal_encoding': 'concat'
    }

    try:
        model_multi = GraphCast(multi_config).to(device)
        x = torch.randn(1, 7, 2, 240, 240).to(device)

        with torch.no_grad():
            y = model_multi(x)

        expected_shape = (1, 3, 2, 240, 240)
        print(f"Multi-frame output shape: {y.shape} (expected {expected_shape})")

        assert y.shape == expected_shape, f"Shape mismatch!"
        print("✓ Multi-frame output test passed!")

    except Exception as e:
        print(f"✗ Multi-frame output test failed!")
        print(f"Error: {str(e)}")
        raise

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)


def test_integration():
    """Test integration with existing model infrastructure"""

    print("\nTesting Integration with Project Infrastructure")
    print("=" * 60)

    # Check if model can be imported alongside other models
    try:
        from models.cnn_model import OceanCNN
        from models.transformer_model import OceanTransformer
        from models.graphcast import GraphCast

        print("✓ Model imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return

    # Test that GraphCast follows same interface as other models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        'input_len': 7,
        'output_len': 1,
        'in_channels': 2,
        'input_res': [240, 240],
        'hidden_dim': 32,
        'mesh_level': 2,
        'processor_layers': 2,
    }

    try:
        # All models should accept the same input format
        x = torch.randn(1, 7, 2, 240, 240).to(device)

        # GraphCast
        model = GraphCast(config).to(device)
        with torch.no_grad():
            y = model(x)

        print(f"✓ GraphCast output shape: {y.shape}")

        # Verify output format matches expected
        assert y.shape == (1, 1, 2, 240, 240)
        print("✓ Integration test passed!")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        raise

    print("=" * 60)


if __name__ == "__main__":
    # Run tests
    test_graphcast_model()
    test_integration()

    print("\n🎉 All GraphCast tests completed successfully!")
    print("The model is ready to be integrated into your training pipeline.")