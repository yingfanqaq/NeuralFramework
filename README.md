# Ocean Velocity Prediction Framework

基于深度学习的海洋速度场预测框架，支持CNN、ResNet和Transformer三种模型架构，具有清晰的数据集配置管理和分离式可视化系统。

## 🚀 主要特性

- **多模型支持**: CNN、ResNet、Transformer三种架构
- **多数据集支持**: 表层海洋数据、中层海洋数据、珠江口数据
- **多GPU训练**: 支持DataParallel和DistributedDataParallel
- **分离式可视化**: 训练与可视化分离，支持独立分析
- **完整Mask支持**: Per-region mask处理，正确的loss计算
- **清晰配置管理**: 不同数据集类型使用专用配置文件
- **完整评估指标**: MSE、MAE、RMSE、R²、MAPE等

## 📁 项目结构

```
NeuralFramework/
├── configs/                    # 配置文件
│   ├── surface_config.yaml     # 表层海洋数据配置
│   ├── mid_config.yaml         # 中层海洋数据配置
│   ├── pearl_river_config.yaml # 珠江口数据配置
│   └── ocean_config.yaml       # 遗留通用配置
├── datasets/                   # 数据集
│   ├── data/                   # 数据文件夹
│   │   ├── uovo_mid_1997-01-01_to_1997-12-31.h5      # 中层数据
│   │   ├── uovo_surface_1997-01-01_to_1997-12-31.h5  # 表层数据
│   │   └── data_pearl_river_estuary_combined.mat     # 珠江口数据
│   ├── base.py                 # 基础数据集类
│   └── ocean_dataset.py        # 海洋数据集类
├── models/                     # 模型定义
│   ├── base.py                 # 基础模型类
│   ├── cnn_model.py            # CNN模型(ConvLSTM)
│   ├── resnet_model.py         # ResNet模型
│   └── transformer_model.py    # Transformer模型
├── trainers/                   # 训练器
│   ├── base.py                 # 基础训练器
│   └── ocean_trainer.py        # 海洋预测训练器
├── procedures/                 # 训练流程
│   ├── base.py                 # 基础流程
│   └── ocean_procedure.py      # 海洋预测流程
├── utils/                      # 工具函数
│   ├── helper.py               # 辅助函数
│   ├── metrics.py              # 评估指标
│   └── visualization.py        # 可视化工具(支持lat/lon)
├── scripts/                    # 运行脚本
│   ├── test.sh                 # 测试脚本
│   └── run_ddp_auto.sh         # DDP训练脚本（自动检测GPU数量）
├── logs/                       # 日志输出目录
├── main.py                     # 主程序入口
├── visualize.py                # 主可视化脚本
├── view_metrics.py             # 查看训练指标脚本
└── README.md                   # 本文档
```

## 📊 数据集说明

### 支持的数据集类型

| 数据集类型    | 配置文件                  | 数据路径       | patches_per_day | 用途           |
| ------------- | ------------------------- | -------------- | --------------- | -------------- |
| `surface`     | `surface_config.yaml`     | 表层数据       | 147             | 表层海洋预测   |
| `mid`         | `mid_config.yaml`         | 中层数据       | 131             | 中层海洋预测   |
| `pearl_river` | `pearl_river_config.yaml` | 珠江口MAT数据  | 1               | 珠江口区域预测 |
| `ocean`       | `ocean_config.yaml`       | 通用（可配置） | 自动检测        | 遗留配置       |

### 数据格式

- **H5格式 (表层/中层)**:
  - 维度: (N, 2, H, W) - N个样本，2个速度分量(u,v)，H×W空间分辨率
  - Mask: (patches_per_day, H, W) - 每个区域的陆地/海洋掩码

- **MAT格式 (珠江口)**:
  - u_combined, v_combined: (time, H, W) - 速度分量
  - 自动从NaN值生成mask
  - 单区域数据 (patches_per_day=1)

- **时间序列**: 支持多步预测，可配置输入和输出长度
- **空间掩码**: 支持陆地/海洋掩码，只对海洋区域计算损失
- **归一化**: 支持标准化和min-max归一化

## ⚙️ 配置管理

### 数据集配置分离

项目采用清晰的数据集配置分离设计：

#### 表层海洋数据配置 (`configs/surface_config.yaml`)
```yaml
data:
  name: 'surface'           # 数据集类型标识
  dataset: 'surface'        # 目录生成标识
  data_path: 'datasets/data/uovo_surface_1997-01-01_to_1997-12-31.h5'
  patches_per_day: 147      # 表层数据特征
```

#### 中层海洋数据配置 (`configs/mid_config.yaml`)
```yaml
data:
  name: 'mid'               # 数据集类型标识
  dataset: 'mid'            # 目录生成标识
  data_path: 'datasets/data/uovo_mid_1997-01-01_to_1997-12-31.h5'
  patches_per_day: 131      # 中层数据特征
```

#### 珠江口数据配置 (`configs/pearl_river_config.yaml`)
```yaml
data:
  name: 'ocean'             # 数据集类型标识
  dataset: 'pearl_river'    # 目录生成标识
  data_path: 'datasets/data/data_pearl_river_estuary_combined.mat'
  patches_per_day: 1        # 单区域数据！

  input_len: 7              # 输入7天
  output_len: 1             # 预测1天
```

### 目录结构生成

使用不同配置会生成不同的目录结构：

```bash
# 表层数据
logs/surface/10_16/OceanCNN_14_30_45/

# 中层数据
logs/mid/10_16/OceanCNN_14_30_45/

# 珠江口数据
logs/pearl_river/10_16/OceanCNN_14_30_45/
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd NeuralFramework

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn h5py pyyaml tqdm
pip install scipy  # 用于MAT文件读取
pip install cartopy  # 可选，用于地理投影可视化
pip install wandb  # 可选，用于实验跟踪
```

### 2. 数据准备

确保数据文件位于正确位置：
```bash
datasets/data/
├── uovo_surface_1997-01-01_to_1997-12-31.h5     # 表层数据
├── uovo_mid_1997-01-01_to_1997-12-31.h5         # 中层数据
└── data_pearl_river_estuary_combined.mat        # 珠江口数据
```

### 3. 训练模型

#### 珠江口数据训练（推荐）
```bash
# 训练
python main.py --config configs/pearl_river_config.yaml --mode train

# 测试并保存预测结果
python main.py --config configs/pearl_river_config.yaml --mode test \
    --model_path logs/pearl_river/10_16/OceanCNN_13_22_08/best_model.pth
```

#### 表层海洋数据训练
```bash
# 训练
python main.py --config configs/surface_config.yaml --mode train

# 测试
python main.py --config configs/surface_config.yaml --mode test
```

#### 中层海洋数据训练
```bash
# 训练
python main.py --config configs/mid_config.yaml --mode train

# 测试
python main.py --config configs/mid_config.yaml --mode test
```

#### 多GPU训练

##### DataParallel (DP) 模式
```bash
# 单进程多GPU训练（自动检测GPU数量）
python main.py --config configs/pearl_river_config.yaml --mode train
```

##### DistributedDataParallel (DDP) 模式
```bash
# 使用自动检测脚本（推荐）
bash scripts/run_ddp_auto.sh --config configs/pearl_river_config.yaml

# 直接使用torchrun
torchrun --nproc_per_node=8 main.py --config configs/pearl_river_config.yaml --mode train
```

**注意**:
- DDP模式必须通过脚本启动，不能直接运行 `python main.py`
- 测试模式建议使用单GPU或DP模式

### 4. 可视化与分析

#### 查看训练指标
```bash
# 查看最终指标（训练后生成的 final_metrics.npz）
python view_metrics.py --model_dir logs/pearl_river/10_16/OceanCNN_13_22_08

# 查看测试指标（测试后生成的 test_metrics.npz）
python view_metrics.py --metrics_file logs/pearl_river/10_16/OceanCNN_13_22_08/test_metrics.npz
```

输出示例：
```
================================================================================
Loading metrics from: logs/pearl_river/10_16/OceanCNN_13_22_08/final_metrics.npz
================================================================================

Available metrics: valid_metrics, test_metrics

Validation Metrics:
------------------------------------------------------------
Loss Metrics:
  valid_loss                = 1.234567e-02

Performance Metrics:
  valid_MAE                 = 0.123456
  valid_RMSE                = 0.234567
  valid_R2                  = 0.876543

Test Metrics:
------------------------------------------------------------
...
```

#### 生成可视化
```bash
# 基本使用 - 可视化前5个样本
python visualize.py --pred_dir logs/pearl_river/10_16/OceanCNN_13_22_08/test_predictions

# 可视化特定样本
python visualize.py --pred_dir logs/pearl_river/10_16/OceanCNN_13_22_08/test_predictions --sample_idx 10

# 可视化时间序列
python visualize.py --pred_dir logs/pearl_river/10_16/OceanCNN_13_22_08/test_predictions --sample_idx 0 --show_sequences

# 只生成统计图
python visualize.py --pred_dir logs/pearl_river/10_16/OceanCNN_13_22_08/test_predictions --stats_only

# 指定输出目录和样本数
python visualize.py --pred_dir logs/pearl_river/10_16/OceanCNN_13_22_08/test_predictions \
    --output_dir my_visualizations --num_samples 10
```

## 🎨 可视化系统

### 分离式可视化设计

项目采用**两阶段可视化**工作流程：

1. **训练/测试阶段**: 只保存预测数据，不进行可视化
2. **可视化阶段**: 使用独立脚本进行可视化分析

### 文件结构

#### 测试后的目录结构
```
logs/pearl_river/10_16/OceanCNN_13_22_08/
├── config.yaml                # 训练配置
├── best_model.pth             # 最佳模型权重
├── final_metrics.npz          # 最终指标（训练模式生成）
├── test_metrics.npz           # 测试指标（测试模式生成）
├── train_rank_0.log           # 训练日志
└── test_predictions/          # 预测数据目录
    ├── all_predictions.npz    # 所有预测数据
    └── metadata.npz           # 元数据（坐标、掩码等）
```

#### 可视化后的目录结构
```
logs/pearl_river/10_16/OceanCNN_13_22_08/test_predictions/
├── all_predictions.npz
├── metadata.npz
└── visualizations/                    # 可视化结果
    ├── comparison_sample_000_t0.png   # 预测vs真实值对比
    ├── input_sequence_sample_000.png  # 输入序列
    ├── prediction_sequence_sample_000.png  # 预测序列
    ├── target_sequence_sample_000.png      # 目标序列
    └── statistics.png                 # 统计信息图
```

### 生成的可视化内容

#### 1. 对比图 (comparison_sample_XXX_tY.png)
- **左侧**: 真实值 (Ground Truth)
- **中间**: 预测值 (Prediction)
- **右侧**: 绝对误差 (Absolute Error)

#### 2. 时间序列图 (可选，使用 --show_sequences)
- **输入序列**: 显示模型输入的时间序列数据
- **预测序列**: 显示模型预测的时间序列
- **目标序列**: 显示真实值的时间序列

#### 3. 统计信息图 (statistics.png)
- **左上**: 误差分布直方图
- **右上**: 预测vs真实值散点图
- **左下**: 每个样本的平均误差曲线
- **右下**: 空间误差热图

### 数据格式说明

#### all_predictions.npz 包含：
```python
{
    'inputs': (N, T_in, C, H, W),      # 输入序列
    'predictions': (N, T_out, C, H, W), # 模型预测
    'targets': (N, T_out, C, H, W),    # 真实值
    'patch_indices': (N,)              # 区域索引
}
```

#### metadata.npz 包含：
```python
{
    'lat': (H, W),                     # 纬度网格
    'lon': (H, W),                     # 经度网格
    'mask': (H, W),                    # 全局掩码
    'mask_per_region': (num_regions, H, W),  # 每个区域的掩码
    'patches_per_day': int,            # 每天的区域数
    'num_samples': int,                # 总样本数
    'normalization_mode': str          # 归一化方法
}
```

## 🔧 高级配置

### GPU配置

#### 单GPU训练
```yaml
train:
  device_ids: [0]  # 使用GPU 0
```

#### 多GPU训练
```yaml
train:
  device_ids: [0, 1, 2, 3]  # 使用4张GPU
  distribute_mode: 'DP'      # DataParallel模式
```

#### 分布式训练 (DDP)
```yaml
train:
  device_ids: [0, 1, 2, 3, 4, 5, 6, 7]  # 使用8张GPU
  distribute_mode: 'DDP'                # DistributedDataParallel模式
  local_rank: 0                         # 本地rank（由torchrun自动设置）
```

**重要配置原则**:
- `device_ids` 必须包含所有要使用的GPU
- `NUM_GPUS` (启动参数) = `len(device_ids)` (配置文件)
- 必须通过 `torchrun` 或脚本启动，不能直接运行 `python main.py`

### 数据配置

#### 数据子集
```yaml
data:
  subset: True              # 启用数据子集
  subset_ratio: 0.01        # 使用1%的数据
```

#### 归一化配置
```yaml
data:
  normalization_mode: 'standardize'  # 标准化
  # normalization_mode: 'normalize'   # min-max归一化
  # normalization_mode: 'none'        # 不归一化
```

#### 批处理配置
```yaml
data:
  train_batchsize: 64       # 训练批大小
  eval_batchsize: 64        # 评估批大小
  num_workers: 4            # 数据加载进程数
  pin_memory: True          # 是否使用pin_memory
```

### 训练配置

#### 优化器配置
```yaml
optimizer:
  optimizer: 'AdamW'        # 优化器类型
  lr: 0.001                 # 学习率
  weight_decay: 0.0001      # 权重衰减
```

#### 学习率调度
```yaml
scheduler:
  scheduler: 'MultiStepLR'  # 调度器类型
  milestones: [40, 70, 90]  # 里程碑
  gamma: 0.5                # 衰减因子
```

#### 训练控制
```yaml
train:
  epochs: 100               # 训练轮数
  patience: 20              # 早停耐心值
  eval_freq: 10             # 评估频率
  saving_best: True         # 保存最佳模型
  saving_checkpoint: True   # 保存检查点
  checkpoint_freq: 20       # 检查点频率
```

## 📈 评估指标

### 支持的指标
- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **R²**: 决定系数
- **MAPE**: 平均绝对百分比误差

### 指标计算
- **训练时**: 只计算loss用于进度监控
- **评估时**: 计算所有指标，基于全局海洋像素累积
- **Per-region mask**: 每个样本使用对应区域的mask
- **正确归一化**: Loss除以实际海洋像素数，不是总像素数

## 🔍 Mask处理机制

### Per-Region Mask支持

项目实现了完整的per-region mask处理机制，确保：

1. **正确的mask选择**: 每个样本根据 `patch_idx` 选择对应区域的mask
2. **正确的loss计算**: 只在海洋像素上计算loss，正确归一化
3. **正确的梯度流**: 陆地区域梯度为0，只更新海洋区域

### Mask数据流

```
数据加载
├── H5: mask shape (patches_per_day, H, W) 或 (N, H, W)
├── MAT: 自动从NaN生成 mask shape (1, H, W)
└── 存储为 mask_per_region: (patches_per_day, H, W)
          ↓
训练循环
├── Batch: (x, y, patch_idx)  # patch_idx shape (B,)
├── 选择mask: batch_masks = mask_per_region[patch_idx]  # (B, H, W)
├── 转为ocean mask: ocean_mask = ~batch_masks  # True=ocean
├── 扩展维度: ocean_mask -> (B, T, C, H, W)
└── 计算loss: loss = ((pred - target)² * ocean_mask).sum() / ocean_mask.sum()
```

### Loss计算修正

**之前（错误）**:
```python
loss = mse_loss(pred * mask, target * mask)  # 除以所有像素
```

**现在（正确）**:
```python
squared_error = (pred - target) ** 2 * ocean_mask
loss = squared_error.sum() / (ocean_mask.sum() + 1e-8)  # 只除以海洋像素
```

**影响**: 如果海洋占30%，loss会正确放大3.3倍，梯度也相应正确缩放。

## 🛠️ 故障排除

### 常见问题

#### 1. 找不到predictions目录
```
FileNotFoundError: Predictions directory not found
```
**解决**: 确保先运行 `--mode test` 生成预测数据

#### 2. torch.load 设备错误
```
TypeError: 'int' object is not callable
```
**解决**: 已修复。如遇到，确保 `trainers/ocean_trainer.py` 的 `_load_model` 方法将 `self.device` 转换为字符串格式

#### 3. 坐标信息缺失
```
No coordinate information found, generating default coordinates
```
**解决**: 这是正常的，可视化脚本会自动生成默认坐标

#### 4. 内存不足
```
MemoryError: Unable to allocate array
```
**解决**: 减少 `--num_samples` 参数或使用数据子集

#### 5. GPU配置问题
```
GPU X not available. Only Y GPUs found.
```
**解决**: 检查 `device_ids` 配置，确保GPU ID有效

#### 6. DDP启动失败
```
DDP setup failed: environment variable RANK expected, but not set
```
**解决**:
- 使用自动检测脚本: `bash scripts/run_ddp_auto.sh --config configs/pearl_river_config.yaml`
- 或使用torchrun: `torchrun --nproc_per_node=8 main.py --config configs/pearl_river_config.yaml --mode train`
- 不要直接运行 `python main.py`

#### 7. patches_per_day配置错误
```
ValueError: patches_per_day must be specified!
```
**解决**: 在配置文件中明确设置：
- Surface数据: `patches_per_day: 147`
- Mid-depth数据: `patches_per_day: 131`
- Pearl River数据: `patches_per_day: 1`

### 调试模式
```bash
# 启用详细日志
export PYTHONPATH=/path/to/NeuralFramework:$PYTHONPATH
python main.py --config configs/pearl_river_config.yaml --mode train
```

## 📚 扩展功能

### 自定义可视化
可以修改 `utils/visualization.py` 中的 `OceanVisualizer` 类来添加：
- 新的可视化类型
- 自定义颜色映射
- 不同的图表样式
- 动画效果

### 批量处理
```bash
# 处理多个模型
for model_dir in logs/pearl_river/*/; do
    python visualize.py --pred_dir "$model_dir/test_predictions"
done
```

### 自定义数据集
1. 在 `datasets/ocean_dataset.py` 中实现数据加载方法
2. 创建对应的配置文件，明确指定 `patches_per_day`
3. 在 `main.py` 中添加新的流程选择逻辑

## 🎯 完整工作流程示例

```bash
# 1. 训练模型
python main.py --config configs/pearl_river_config.yaml --mode train

# 2. 测试模型并保存预测
python main.py --config configs/pearl_river_config.yaml --mode test \
    --model_path logs/pearl_river/10_16/OceanCNN_13_22_08/best_model.pth

# 3. 查看训练指标
python view_metrics.py --model_dir logs/pearl_river/10_16/OceanCNN_13_22_08

# 4. 生成可视化
python visualize.py --pred_dir logs/pearl_river/10_16/OceanCNN_13_22_08/test_predictions \
    --num_samples 10 --show_sequences

# 5. 查看结果
ls logs/pearl_river/10_16/OceanCNN_13_22_08/test_predictions/visualizations/
```
