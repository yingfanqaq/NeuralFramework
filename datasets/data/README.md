# 海洋流场数据 MAT 文件

## 概述

本 MAT 文件包含珠江口地区经过处理的海洋流场和海表高度数据，时间覆盖 1993年1月1日 至 1997年12月31日。

## 数据来源

- **原始数据**: 哥白尼海洋服务全球海洋物理分析和预报产品
- **数据集ID**: `cmems_mod_glo_phy_my_0.083deg_P1D-m`
- **空间分辨率**: 1/12° (~8.3 公里)
- **时间分辨率**: 日数据
- **研究区域**: 珠江口 (103.5°E-123.5°E, 12.5°N-32.5°N)

## 文件结构

### 变量说明

| 变量名 | 维度 | 描述 | 单位 |
|--------|------|------|------|
| `u_combined` | (time, 240, 240) | 东向流速分量 | m/s |
| `v_combined` | (time, 240, 240) | 北向流速分量 | m/s |
| `adt_combined` | (time, 240, 240) | 海表高度异常 | m |
| `time_combined` | (time,) | 时间数组 (MATLAB datenum 格式) | 天 |
| `x` | (240, 240) | 经度网格 | 度 |
| `y` | (240, 240) | 纬度网格 | 度 |
| `pm` | (240, 240) | 网格间距倒数 (1/dx) | 1/m |
| `pn` | (240, 240) | 网格间距倒数 (1/dy) | 1/m |
| `f` | (240, 240) | 科里奥利参数 | 1/s |

### 时间覆盖

- **开始日期**: 1993年1月1日
- **结束日期**: 1997年12月31日
- **总天数**: 1826 天 (5年，包含闰年1996)
- **时间格式**: MATLAB datenum (自公元0年1月1日以来的天数)

### 空间覆盖

- **网格大小**: 240 × 240 个格点
- **经度范围**: 103.5°E 至 123.5°E
- **纬度范围**: 12.5°N 至 32.5°N
- **研究区域**: 珠江口及周边南海海域

## 使用示例

### MATLAB
```matlab
% 加载数据
data = load('data_pearl_river_estuary_combined.mat');
u = data.u_combined;
v = data.v_combined;
h = data.adt_combined;
t = data.time_combined;
x = data.x';
y = data.y';
pm = data.pm';
pn = data.pn';
f = data.f';

% 转换时间格式
dates = datetime(t, 'ConvertFrom', 'datenum');

% 绘制第一个时间步
figure;
pcolor(x, y, squeeze(u(1,:,:)));
shading interp;
colorbar;
title('海表流场东向分量 (1993-01-01)');
xlabel('经度 (°E)');
ylabel('纬度 (°N)');
```
```python
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 加载数据
data = loadmat('data_pearl_river_estuary_combined.mat')
u, v, h, t = data['u_combined'], data['v_combined'], data['adt_combined'], data['time_combined']
x, y, pm, pn, f = data['x'].T, data['y'].T, data['pm'].T, data['pn'].T, data['f'].T

# 转换时间格式
def matlab_datenum_to_datetime(datenum):
    return datetime(1, 1, 1) + timedelta(days=int(datenum) - 1)

dates = [matlab_datenum_to_datetime(t_val) for t_val in t.flatten()]

# 绘制第一个时间步
plt.figure(figsize=(12, 8))
plt.pcolormesh(x, y, u[0], shading='auto', cmap='RdBu_r')
plt.colorbar(label='流速 (m/s)')
plt.title('海表流场东向分量 (1993-01-01)')
plt.xlabel('经度 (°E)')
plt.ylabel('纬度 (°N)')
plt.show()

# 计算流速大小
speed = np.sqrt(u**2 + v**2)
print(f"流速范围: {np.nanmin(speed):.3f} - {np.nanmax(speed):.3f} m/s")
```
## 数据处理流程
1. 数据下载: 从哥白尼海洋服务下载原始 NetCDF 数据
2. 区域裁剪: 提取珠江口及周边海域 (240×240 网格)
3. 时间对齐: 确保 u、v、海表高度数据时间维度一致
4. 参数计算: 计算网格参数 pm、pn 和科里奥利参数 f
5. 格式转换: 转换为 MATLAB 兼容的 MAT 文件格式