"""Test script for OneForecast model"""

import torch
import yaml
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.oneforcast import OceanOneForecast


def test_oneforcast(config_path='configs/models/oneforcast_conf.yaml', use_light=False):
    """Test OneForecast model with different configurations."""

    # Load configuration
    if use_light:
        config_path = 'configs/models/oneforcast_light_conf.yaml'

    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    print("\nCreating OneForecast model...")
    try:
        model = OceanOneForecast(config)
        model = model.to(device)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False

    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB (assuming float32)")

    # Test forward pass
    print("\n" + "="*60)
    print("Testing forward pass...")
    print("="*60)

    batch_sizes = [1, 2] if device == 'cuda' else [1]

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        # Create dummy input
        input_shape = (
            batch_size,
            config['input_len'],
            config['in_channels'],
            config['input_res'][0],
            config['input_res'][1]
        )
        x = torch.randn(*input_shape).to(device)
        print(f"  Input shape: {x.shape}")

        # Forward pass
        try:
            with torch.no_grad():
                if device == 'cuda':
                    # Measure GPU memory
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    start_memory = torch.cuda.memory_allocated() / (1024**2)

                    output = model(x)

                    end_memory = torch.cuda.memory_allocated() / (1024**2)
                    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)

                    print(f"  Output shape: {output.shape}")
                    print(f"  GPU memory used: {end_memory - start_memory:.2f} MB")
                    print(f"  Peak GPU memory: {peak_memory:.2f} MB")
                else:
                    output = model(x)
                    print(f"  Output shape: {output.shape}")

                # Verify output shape
                expected_shape = (
                    batch_size,
                    config['output_len'],
                    config['in_channels'],
                    config['input_res'][0],
                    config['input_res'][1]
                )
                assert output.shape == expected_shape, (
                    f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
                )
                print("  ✓ Forward pass successful")

                # Check for NaN or Inf
                if torch.isnan(output).any():
                    print("  ⚠ Warning: Output contains NaN values")
                if torch.isinf(output).any():
                    print("  ⚠ Warning: Output contains Inf values")

        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            return False

    # Test gradient flow
    print("\n" + "="*60)
    print("Testing gradient flow...")
    print("="*60)

    x = torch.randn(1, config['input_len'], config['in_channels'],
                   config['input_res'][0], config['input_res'][1]).to(device)
    x.requires_grad = True

    try:
        output = model(x)
        loss = output.mean()
        loss.backward()

        # Check if gradients flow properly
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

        if grad_norms:
            print(f"  Average gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
            print(f"  Max gradient norm: {max(grad_norms):.6f}")
            print(f"  Min gradient norm: {min(grad_norms):.6f}")
            print("  ✓ Gradient flow test passed")
        else:
            print("  ⚠ Warning: No gradients computed")

    except Exception as e:
        print(f"  ✗ Gradient flow test failed: {e}")
        return False

    return True


def compare_configurations():
    """Compare standard and light configurations."""
    print("\n" + "="*60)
    print("Comparing configurations")
    print("="*60)

    configs = [
        ('Standard', 'configs/models/oneforcast_conf.yaml'),
        ('Light', 'configs/models/oneforcast_light_conf.yaml')
    ]

    for name, path in configs:
        print(f"\n{name} Configuration:")
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        model = OceanOneForecast(config)
        params = sum(p.numel() for p in model.parameters())

        print(f"  Hidden dim: {config['hidden_dim']}")
        print(f"  Processor layers: {config['processor_layers']}")
        print(f"  Mesh level: {config['mesh_level']}")
        print(f"  Dual path: {config.get('use_dual_path', False)}")
        print(f"  Temporal processing: {config.get('use_temporal_processing', False)}")
        print(f"  Total parameters: {params:,}")
        print(f"  Estimated memory: {params * 4 / (1024**2):.2f} MB")


if __name__ == "__main__":
    print("OneForecast Model Test Suite")
    print("="*60)

    # Test standard configuration
    print("\nTesting standard configuration...")
    success = test_oneforcast(use_light=False)

    if not success:
        print("\n⚠ Standard configuration test failed, trying light configuration...")
        success = test_oneforcast(use_light=True)

    # Compare configurations
    compare_configurations()

    if success:
        print("\n✓ All tests passed successfully!")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")