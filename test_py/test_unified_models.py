"""
Test script to verify unified model naming and autoregressive versions
"""

import torch
import yaml
import sys

# Add parent directory to path
sys.path.append('.')

def test_model_imports():
    """Test that all renamed models can be imported"""
    print("Testing model imports...")
    print("=" * 60)

    try:
        # Import base models
        from models.fuxi import Fuxi, FuxiAutoregressive
        from models.nng import NNG, NNGAutoregressive
        from models.oneforcast_dp import OneForecast, OneForecastAutoregressive
        from models.graphcast import GraphCast, GraphCastAutoregressive

        print("✓ All model imports successful")

        # Check backward compatibility
        from models import _model_dict

        # Test new names
        assert "Fuxi" in _model_dict
        assert "NNG" in _model_dict
        assert "OneForecast" in _model_dict
        assert "GraphCast" in _model_dict

        # Test autoregressive versions
        assert "Fuxi_Auto" in _model_dict
        assert "NNG_Auto" in _model_dict
        assert "OneForecast_Auto" in _model_dict
        assert "GraphCast_Auto" in _model_dict

        # Test backward compatibility
        assert "OceanFuxi" in _model_dict
        assert "OceanNNG" in _model_dict
        assert "OceanOneForecast" in _model_dict

        print("✓ Model registry updated correctly")
        print("✓ Backward compatibility maintained")

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False

    return True


def test_model_instantiation():
    """Test that models can be instantiated with test configs"""
    print("\nTesting model instantiation...")
    print("=" * 60)

    # Test configurations
    test_configs = {
        "Fuxi": {
            'input_len': 7,
            'output_len': 1,
            'in_channels': 2,
            'img_size': [240, 240],
            'patch_size': [4, 4],
            'embed_dim': 128,  # Reduced for testing
            'num_groups': 8,
            'num_heads': 4,
            'window_size': 7,
            'depth': 2,  # Reduced for testing
            'use_3d_path': False,
            'pseudo_depth': 2
        },
        "NNG": {
            'input_len': 7,
            'output_len': 1,
            'in_channels': 2,
            'input_res': [240, 240],
            'mesh_level': 2,  # Reduced for testing
            'multimesh': False,
            'hidden_dim': 32,  # Reduced for testing
            'processor_layers': 2,  # Reduced for testing
            'aggregation': 'sum',
            'use_cugraphops': False,
            'add_3d_dim': False
        },
        "OneForecast": {
            'input_len': 7,
            'output_len': 1,
            'in_channels': 2,
            'input_res': [240, 240],
            'hidden_dim': 64,  # Reduced for testing
            'num_layers': 1,
            'processor_layers': 2,  # Reduced for testing
            'mesh_level': 2,  # Reduced for testing
            'multimesh': False
        },
        "GraphCast": {
            'input_len': 7,
            'output_len': 1,
            'in_channels': 2,
            'input_res': [240, 240],
            'hidden_dim': 32,  # Reduced for testing
            'mesh_level': 2,  # Reduced for testing
            'processor_layers': 2,  # Reduced for testing
            'add_3d_dim': False,
            'temporal_encoding': 'concat'
        }
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name, config in test_configs.items():
        try:
            print(f"\nTesting {model_name}...")

            # Import model class
            if model_name == "Fuxi":
                from models.fuxi import Fuxi, FuxiAutoregressive
                model = Fuxi(config)
                model_auto = FuxiAutoregressive(config)
            elif model_name == "NNG":
                from models.nng import NNG, NNGAutoregressive
                model = NNG(config)
                model_auto = NNGAutoregressive(config)
            elif model_name == "OneForecast":
                from models.oneforcast_dp import OneForecast, OneForecastAutoregressive
                model = OneForecast(config)
                model_auto = OneForecastAutoregressive(config)
            elif model_name == "GraphCast":
                from models.graphcast import GraphCast, GraphCastAutoregressive
                model = GraphCast(config)
                model_auto = GraphCastAutoregressive(config)

            # Move to device
            model = model.to(device)
            model_auto = model_auto.to(device)

            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            params_auto = sum(p.numel() for p in model_auto.parameters())

            print(f"  Standard model parameters: {params:,}")
            print(f"  Autoregressive parameters: {params_auto:,}")

            # Test forward pass
            x = torch.randn(1, config['input_len'], config['in_channels'],
                          config.get('input_res', config.get('img_size'))[0],
                          config.get('input_res', config.get('img_size'))[1]).to(device)

            with torch.no_grad():
                # Test standard model
                y = model(x)
                assert y.shape == (1, config['output_len'], config['in_channels'],
                                 config.get('input_res', config.get('img_size'))[0],
                                 config.get('input_res', config.get('img_size'))[1])

                # Test autoregressive model
                y_auto = model_auto(x, rollout_steps=3)
                assert y_auto.shape == (1, 3, config['in_channels'],
                                       config.get('input_res', config.get('img_size'))[0],
                                       config.get('input_res', config.get('img_size'))[1])

            print(f"  ✓ Forward pass successful")
            print(f"  ✓ Autoregressive mode working")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False

    return True


def test_config_mapping():
    """Test that config mappings are updated correctly"""
    print("\nTesting config mappings...")
    print("=" * 60)

    try:
        with open('configs/models/model_mapping.yaml', 'r') as f:
            mappings = yaml.safe_load(f)

        # Check new model names
        new_models = [
            "Fuxi", "Fuxi_Light", "Fuxi_Full", "Fuxi_Auto",
            "NNG", "NNG_Light", "NNG_Full", "NNG_Auto",
            "OneForecast", "OneForecast_Light", "OneForecast_Balanced", "OneForecast_Auto",
            "GraphCast", "GraphCast_Light", "GraphCast_Heavy", "GraphCast_Auto"
        ]

        for model in new_models:
            assert model in mappings, f"Missing mapping for {model}"

        # Check backward compatibility
        old_models = [
            "OceanFuxi", "OceanFuxi_Light", "OceanFuxi_Full",
            "OceanNNG", "OceanNNG_Light", "OceanNNG_Full",
            "OceanOneForecast", "OceanOneForecast_Light", "OceanOneForecast_Balanced"
        ]

        for model in old_models:
            assert model in mappings, f"Missing backward compatibility mapping for {model}"

        print("✓ All config mappings present")
        print("✓ Backward compatibility maintained")

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("UNIFIED MODEL NAMING TEST SUITE")
    print("=" * 60)

    all_passed = True

    # Run tests
    all_passed &= test_model_imports()
    all_passed &= test_model_instantiation()
    all_passed &= test_config_mapping()

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nSummary of changes:")
        print("1. Renamed models (removed 'Ocean' prefix):")
        print("   - OceanFuxi → Fuxi")
        print("   - OceanNNG → NNG")
        print("   - OceanOneForecast → OneForecast")
        print("\n2. Added autoregressive versions:")
        print("   - FuxiAutoregressive (Fuxi_Auto)")
        print("   - NNGAutoregressive (NNG_Auto)")
        print("   - OneForecastAutoregressive (OneForecast_Auto)")
        print("   - GraphCastAutoregressive (GraphCast_Auto)")
        print("\n3. Maintained backward compatibility:")
        print("   - Old names still work (deprecated)")
        print("\n4. Updated configuration mappings")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()