"""
Test script for Crossformer model
Verifies model initialization and forward pass
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crossformer import OceanCrossformer, OceanCrossformerAutoregressive
import yaml


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_variant(config_path, model_class, variant_name):
    """Test a specific model variant"""
    print(f"\n{'='*60}")
    print(f"Testing {variant_name}")
    print(f"{'='*60}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model
    model = model_class(config)
    model.eval()
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"\nModel Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 2
    input_len = config['input_len']
    in_channels = config['in_channels']
    img_size = config['img_size']
    
    x = torch.randn(batch_size, input_len, in_channels, img_size[0], img_size[1])
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        if 'Auto' in variant_name:
            # Test autoregressive
            rollout_steps = config.get('rollout_steps', 3)
            output = model(x, rollout_steps=rollout_steps)
            expected_output_len = rollout_steps
        else:
            # Test single-step
            output = model(x)
            expected_output_len = config['output_len']
    
    print(f"Output shape: {output.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, expected_output_len, in_channels, img_size[0], img_size[1])
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print(f"✓ Output shape correct: {output.shape}")
    print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Check for NaN or Inf
    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"
    print(f"✓ No NaN or Inf values")
    
    return num_params


def main():
    """Run all tests"""
    print("="*60)
    print("Crossformer Model Tests")
    print("="*60)
    
    # Test variants
    variants = [
        ('configs/models/crossformer_light_conf.yaml', OceanCrossformer, 'Crossformer_Light'),
        ('configs/models/crossformer_balanced_conf.yaml', OceanCrossformer, 'Crossformer_Balanced'),
        ('configs/models/crossformer_full_conf.yaml', OceanCrossformer, 'Crossformer_Full'),
        ('configs/models/crossformer_auto_conf.yaml', OceanCrossformerAutoregressive, 'Crossformer_Auto'),
    ]
    
    results = {}
    
    for config_path, model_class, variant_name in variants:
        try:
            num_params = test_model_variant(config_path, model_class, variant_name)
            results[variant_name] = ('✓ PASSED', num_params)
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[variant_name] = ('✗ FAILED', 0)
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for variant, (status, num_params) in results.items():
        if num_params > 0:
            print(f"{variant:25s} {status:10s} {num_params/1e6:8.2f}M params")
        else:
            print(f"{variant:25s} {status:10s}")
    
    # Overall result
    all_passed = all(status == '✓ PASSED' for status, _ in results.values())
    if all_passed:
        print(f"\n{'='*60}")
        print("✓ ALL TESTS PASSED!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("✗ SOME TESTS FAILED")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == '__main__':
    main()

