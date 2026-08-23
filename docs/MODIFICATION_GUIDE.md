# 湍流数据生成程序修改说明

## 📋 修改概要

本文档说明了对湍流数据生成程序 `generate_turbulence_1000_tuned.py` 的主要修改。

---

## 🎯 主要修改内容

### 1. **输入参数改变**
- **原始版本**: 使用 `u_rms`（速度脉动RMS）作为输入
- **新版本**: 使用 **TKE（湍动能）** 和 **epsilon（耗散率）** 作为输入

**关系式**:
```
u_prime = sqrt(2 * TKE / 3)
L = u_prime³ / epsilon
```

### 2. **网格参数调整**
```python
# 原始参数
L = 0.72 * π        # 域长度
N = 1024            # 网格点数

# 新参数
L = 0.18 * π        # 域长度 [m]
N = 256             # 网格点数
```

### 3. **参数范围和样本数**
```python
# 参数范围
TKE范围:     [0.01, 1.0] m²/s²     (对数空间均匀分布)
Epsilon范围:  [0.01, 5.0] m²/s³     (对数空间均匀分布)

# 样本配置
TKE样本数:           10
Epsilon样本数:       10
每组参数随机种子数:   10

# 总样本数
总计: 10 × 10 × 10 = 1000 个样本
```

### 4. **调谐方程实现**

新版本为**每一组(TKE, epsilon)**参数都求解调谐方程，计算对应的能谱系数：

```python
def solve_tuning_equation(tke, epsilon, nu):
    """
    求解调谐方程得到:
    - new_alpha: 调谐后的缩放系数
    - new_kappa_e: 调谐后的能量峰值波数
    """
    # 计算湍流雷诺数
    Re_L = tke² / (epsilon * nu)
    
    # 求解调谐方程 (Eq.10)
    # z * U(7/2, 5/3, z) / U(5/2, 2/3, z) = (2/5) * Re_L^(-1/2)
    
    # 计算调谐系数 (Eq.11-12)
    new_alpha = 4 / (sqrt(π) * U(5/2, 2/3, z))
    new_kappa_e = kappa_eta * sqrt(z / 2)
```

其中 U 是合流超几何函数（confluent hypergeometric function）。

### 5. **能谱函数**

使用调谐后的参数生成VKP能谱：

```python
def von_karman_pao_tuned(k, alpha, kappa_e, u_prime, kappa_eta):
    """调谐后的Von Kármán-Pao能谱"""
    term1 = alpha * (u_prime² / kappa_e)
    term2 = (k / kappa_e)⁴
    term3 = (1 + (k / kappa_e)²)^(-17/6)
    exp_term = exp(-2 * (k / kappa_eta)²)
    return term1 * term2 * term3 * exp_term
```

---

## 📊 数据组织结构

生成的数据按照以下方式组织：

```
TurbulenceData_1000_Tuned/
├── velocity_fields/          # 速度场数据
│   ├── field_0000.npy       # Sample 0 (Param 0, Seed 0)
│   ├── field_0001.npy       # Sample 1 (Param 0, Seed 1)
│   ├── ...
│   ├── field_0010.npy       # Sample 10 (Param 1, Seed 0)
│   └── field_0999.npy       # Sample 999 (Param 99, Seed 9)
│
├── spectra_plots/            # 能谱对比图（每100个样本）
│   ├── spectrum_0000.png
│   ├── spectrum_0100.png
│   └── ...
│
└── metadata/                 # 元数据
    ├── generation_metadata.json   # 生成信息
    └── training_config.json       # 训练配置
```

### 样本编号规则
```
sample_id = param_id * 10 + seed_offset

其中:
- param_id: 0-99 (100组参数组合)
- seed_offset: 0-9 (每组参数10个随机种子)
```

---

## 💾 保存的元数据

### generation_metadata.json 包含:
- 生成时间戳和耗时
- 成功/失败样本统计
- 每个样本的详细信息:
  - TKE, epsilon 输入参数
  - 调谐后的 alpha, kappa_e
  - 实际能量值
  - 文件路径

### training_config.json 包含:
- 数据维度: [256, 256, 2]
- 输入参数类型和范围
- 调谐方法说明

---

## 🚀 使用方法

### 1. 运行生成程序
```bash
python generate_turbulence_1000_tuned.py
```

### 2. 生成过程
程序会:
1. 生成100组(TKE, epsilon)参数组合（对数空间均匀分布）
2. 为每组参数求解调谐方程，获得 alpha 和 kappa_e
3. 为每组参数生成10个不同随机种子的湍流场
4. 每100个样本保存一次能谱对比图
5. 保存所有元数据

### 3. 预期运行时间
- 单个样本: ~1-2秒（取决于CPU）
- 1000个样本总计: ~30-60分钟（使用8核并行）

---

## 📈 关键物理参数

### 固定参数
```python
nu = 1e-5 m²/s          # 分子粘度（空气，20°C）
L = 0.18 * π m          # 域长度
N = 256                 # 网格点数
M = 10000               # 傅里叶模态数
```

### 变化参数
```python
TKE ∈ [0.01, 1.0] m²/s²      # 湍动能
epsilon ∈ [0.01, 5.0] m²/s³  # 耗散率
```

### 派生参数（调谐方程求解）
```python
u_prime = sqrt(2*TKE/3)           # RMS速度
L = u_prime³/epsilon              # 积分长度尺度
kappa_eta = (epsilon/nu³)^(1/4)  # Kolmogorov波数
Re_L = TKE²/(epsilon*nu)          # 湍流雷诺数
alpha = f(Re_L)                   # 调谐后的缩放系数
kappa_e = f(Re_L, kappa_eta)      # 调谐后的能量峰值波数
```

---

## 🔍 验证建议

生成数据后，建议进行以下验证：

### 1. 能量验证
```python
# 读取速度场
u_field = np.load('field_0000.npy')

# 计算实际能量
energy = 0.5 * np.mean(u_field[..., 0]**2 + u_field[..., 1]**2)

# 与输入TKE比较
# energy 应该接近 TKE
```

### 2. 能谱验证
查看生成的能谱图 `spectrum_*.png`，确认：
- 数值能谱与理论VKP能谱吻合良好
- 峰值位置接近 kappa_e
- 高波数区域有正确的指数衰减

### 3. 参数分布验证
```python
import json

# 读取元数据
with open('metadata/generation_metadata.json', 'r') as f:
    data = json.load(f)

# 检查TKE和epsilon的分布
tkes = [s['tke'] for s in data['successful_samples']]
epsilons = [s['epsilon'] for s in data['successful_samples']]

# 应该在对数空间均匀分布
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(np.log10(tkes), bins=20)
plt.xlabel('log10(TKE)')
plt.subplot(1, 2, 2)
plt.hist(np.log10(epsilons), bins=20)
plt.xlabel('log10(epsilon)')
plt.show()
```

---

## 🔧 配置参数调整

如需修改生成参数，编辑 `Config` 类：

```python
class Config:
    # 修改参数范围
    TKE_MIN = 0.01
    TKE_MAX = 1.0
    EPS_MIN = 0.01
    EPS_MAX = 5.0
    
    # 修改样本数
    N_TKE_SAMPLES = 10
    N_EPS_SAMPLES = 10
    N_SEEDS_PER_PARAM = 10
    
    # 修改网格参数
    L = 0.18 * np.pi
    N = 256
    
    # 修改并行数（根据CPU核心数）
    N_PARALLEL = 8
```

---

## 📝 与原版本的主要区别总结

| 特性 | 原版本 | 新版本 |
|-----|--------|--------|
| 输入参数 | u_rms | TKE + epsilon |
| 网格尺寸 | 1024³ | 256² |
| 域长度 | 0.72π | 0.18π |
| 样本数 | 1500 | 1000 |
| 能谱系数 | 固定 | 每组参数调谐 |
| 参数组合 | 单一参数 | 100组参数组合 |
| 随机种子 | 1500个 | 每组10个 |

---

## ⚠️ 注意事项

1. **内存需求**: 256×256×2 的双精度数组约需 1 MB，1000个样本约 1 GB
2. **CPU使用**: 建议根据CPU核心数调整 `N_PARALLEL` 参数
3. **调谐方程求解**: 某些极端参数组合可能导致求解失败，程序会使用默认值并输出警告
4. **数值稳定性**: 超几何函数在某些参数范围可能不稳定，已添加异常处理

---

## 📧 后续步骤

生成数据后，可以：
1. 使用元数据分析参数分布和能量统计
2. 可视化不同参数组合的湍流场特征
3. 训练神经网络模型（CVAE等）
4. 进行湍流统计分析

---

**创建日期**: 2025-11-05
**版本**: 1.0
