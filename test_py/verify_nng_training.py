#!/usr/bin/env python
"""
Final verification script for OceanNNG training setup
"""

import torch
import yaml
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_training_setup():
    """Verify that OceanNNG is properly configured for training"""

    print("=" * 60)
    print("OceanNNG Training Setup Verification")
    print("=" * 60)

    # 1. Check configuration file
    print("\n1. Checking pearl_river_config.yaml...")
    config_path = 'configs/pearl_river_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    train_batch = config['data']['train_batchsize']
    eval_batch = config['data']['eval_batchsize']

    print(f"   Model: {model_name}")
    print(f"   Train batch size: {train_batch}")
    print(f"   Eval batch size: {eval_batch}")

    if model_name not in ['OceanNNG', 'OceanNNG_Light', 'OceanNNG_Full']:
        print(f"   ⚠️  Model is not OceanNNG. Current: {model_name}")
        return False

    # 2. Check model mapping
    print("\n2. Checking model mapping...")
    mapping_path = 'configs/models/model_mapping.yaml'
    with open(mapping_path, 'r') as f:
        mapping = yaml.safe_load(f)

    if model_name in mapping:
        model_config_path = mapping[model_name]
        print(f"   ✓ {model_name} -> {model_config_path}")
    else:
        print(f"   ✗ {model_name} not found in model_mapping.yaml")
        return False

    # 3. Load model configuration
    print("\n3. Loading model configuration...")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    print(f"   Configuration details:")
    print(f"   - mesh_level: {model_config['mesh_level']}")
    print(f"   - hidden_dim: {model_config['hidden_dim']}")
    print(f"   - processor_layers: {model_config['processor_layers']}")

    # 4. Test model creation
    print("\n4. Testing model creation...")
    try:
        from models import _model_dict
        model_class = _model_dict[model_name]
        model = model_class(model_config)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model created successfully")
        print(f"   Total parameters: {total_params:,}")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False

    # 5. Memory estimation
    print("\n5. Estimated memory requirements...")

    # Based on our tests
    memory_estimates = {
        'OceanNNG_Light': (2.46, 4.79, 9.44),  # batch 1, 2, 4
        'OceanNNG': (9.44, 13.11, 'OOM'),      # batch 1, 2, 4  (balanced)
        'OceanNNG_Full': (21.99, 'OOM', 'OOM') # batch 1, 2, 4
    }

    if model_name in memory_estimates:
        estimates = memory_estimates[model_name]
        print(f"   Memory usage for {model_name}:")
        for i, (batch, mem) in enumerate(zip([1, 2, 4], estimates)):
            if mem != 'OOM':
                status = "✓" if batch <= train_batch else "⚠️"
                print(f"   {status} Batch size {batch}: ~{mem:.2f} GB")
            else:
                print(f"   ✗ Batch size {batch}: Out of Memory")

    # 6. Recommendations
    print(f"\n6. Recommendations:")

    if model_name == 'OceanNNG_Light':
        if train_batch > 4:
            print(f"   ⚠️  Batch size {train_batch} might be too large. Recommended: 4 or less")
        else:
            print(f"   ✓ Batch size {train_batch} is appropriate for lightweight model")
    elif model_name == 'OceanNNG':
        if train_batch > 2:
            print(f"   ⚠️  Batch size {train_batch} might be too large. Recommended: 2 or less")
        else:
            print(f"   ✓ Batch size {train_batch} is appropriate for balanced model")
    elif model_name == 'OceanNNG_Full':
        if train_batch > 1:
            print(f"   ⚠️  Batch size {train_batch} might be too large. Recommended: 1")
        else:
            print(f"   ✓ Batch size {train_batch} is appropriate for full model")

    # 7. Final check
    print(f"\n{'='*60}")
    print("Final Status")
    print(f"{'='*60}")

    can_train = True
    warnings = []

    # Check batch size compatibility
    if model_name == 'OceanNNG_Light' and train_batch > 4:
        warnings.append("Batch size might cause OOM errors")
    elif model_name == 'OceanNNG' and train_batch > 2:
        warnings.append("Batch size might cause OOM errors")
    elif model_name == 'OceanNNG_Full' and train_batch > 1:
        warnings.append("Batch size might cause OOM errors")

    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"   - {warning}")

    if can_train:
        print(f"\n✅ OceanNNG is ready for training!")
        print(f"\n🚀 To start training, run:")
        print(f"   python train.py --config configs/pearl_river_config.yaml")

        print(f"\n💡 Tips:")
        print(f"   - Monitor GPU memory usage during training")
        print(f"   - If OOM occurs, reduce batch size or use OceanNNG_Light")
        print(f"   - For best performance with available memory, use OceanNNG (balanced)")

        return True
    else:
        print(f"\n✗ Setup verification failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    # Use GPU 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    success = verify_training_setup()

    if not success:
        sys.exit(1)