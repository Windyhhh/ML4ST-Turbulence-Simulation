# FNO湍流生成模型 - 参数空间（条件输入 c）设计方案

## 📊 数据集概览

**生成的样本数据**：
- **总样本数**：1000个湍流场
- **参数组合**：100组（10个TKE值 × 10个epsilon值）
- **每组重复**：10个随机种子
- **网格尺寸**：256 × 256
- **数据类型**：2D速度场 (u_x, u_y)

---

## 🎯 核心物理参数

### 1. 直接控制参数（Primary Parameters）

| 参数名称 | 符号 | 物理意义 | 范围 | 单位 | 样本点数 |
|---------|------|---------|------|------|---------|
| **湍动能** | TKE (k) | 湍流脉动动能 | [0.046875, 0.1875] | m²/s² | 10 |
| **耗散率** | epsilon (ε) | 能量耗散率 | [0.2701, 1.0804] | m²/s³ | 10 |

**参数说明**：
- TKE范围：0.5× ~ 2.0× 基准值（0.09375）
- Epsilon范围：0.5× ~ 2.0× 基准值（0.5402）
- 参数采样：线性均匀采样
- 组合方式：笛卡尔积（全组合）

---

### 2. 衍生参数（Derived Parameters）

以下参数由TKE和epsilon通过物理关系计算得出：

| 参数名称 | 符号 | 计算公式 | 物理意义 | 单位 |
|---------|------|---------|---------|------|
| **调谐缩放系数** | α (alpha) | 调谐方程求解 | 能谱幅值 | 无量纲 |
| **能量峰值波数** | κ_e (kappa_e) | κ_η·√(z/2) | 能量谱峰值位置 | m⁻¹ |
| **RMS速度** | u' (u_prime) | √(2k/3) | 湍流强度 | m/s |
| **Kolmogorov波数** | κ_η (kappa_eta) | (ε/ν³)^(1/4) | 最小耗散尺度 | m⁻¹ |
| **湍流雷诺数** | Re_L | k²/(ε·ν) | 湍流强度 | 无量纲 |

**注意**：
- α和κ_e通过求解调谐方程获得（文献方法）
- 调谐方程：z·U(7/2, 5/3, z)/U(5/2, 2/3, z) = (2/5)·Re_L^(-0.5)
- 这些参数已经计算并保存在元数据中

---

### 3. 固定参数（Fixed Parameters）

| 参数名称 | 符号 | 值 | 说明 |
|---------|------|-----|------|
| 域长度 | L | 0.18π ≈ 0.5655 m | 计算域尺寸 |
| 网格点数 | N | 256 | 空间分辨率 |
| 傅里叶模态数 | M | 10000 | 随机模态数量 |
| 分子粘度 | ν | 1×10⁻⁵ m²/s | 空气粘度 |

---

## 💡 条件输入 c 的三种方案

### 方案A：最小条件输入（推荐起步）✅

```python
c = [TKE, epsilon]  # 维度: 2
```

**优点**：
- ✅ 最简洁，只包含物理上独立的参数
- ✅ 所有其他参数都可以从这两个推导
- ✅ 减少模型输入维度
- ✅ 物理意义最清晰

**缺点**：
- ❌ FNO需要隐式学习TKE→α, κ_e的复杂映射
- ❌ 调谐方程求解的复杂性需要网络自己学习

**适用场景**：
- 初步验证FNO的学习能力
- 研究网络能否自主学习物理关系
- 计算资源有限时

**数据准备**：
```python
# 伪代码
condition = np.array([tke, epsilon])  # shape: (2,)
```

---

### 方案B：扩展条件输入（推荐最终）⭐⭐⭐

```python
c = [TKE, epsilon, alpha, kappa_e]  # 维度: 4
```

**优点**：
- ✅ 包含了控制能谱形状的关键参数
- ✅ α直接控制能谱幅值，κ_e控制峰值位置
- ✅ 减轻网络学习负担，加速收敛
- ✅ 物理意义仍然清晰
- ✅ 性能和复杂度的良好平衡

**缺点**：
- ⚠️ 需要从元数据中提取α和κ_e
- ⚠️ 维度略高（但仍可控）

**适用场景**：
- **推荐作为最终方案**
- 需要较好的生成质量
- 有充足的训练数据

**数据准备**：
```python
# 伪代码
# 从metadata中读取
metadata = json.load('generation_metadata.json')
for sample in metadata['successful_samples']:
    condition = np.array([
        sample['tke'],
        sample['epsilon'],
        sample['alpha'],
        sample['kappa_e']
    ])  # shape: (4,)
```

---

### 方案C：完整条件输入（实验用）🔬

```python
c = [TKE, epsilon, alpha, kappa_e, u_prime, Re_L]  # 维度: 6
```

**优点**：
- ✅ 提供最完整的物理信息
- ✅ 可用于消融实验
- ✅ 研究不同参数的重要性

**缺点**：
- ❌ 参数间存在冗余（u', Re_L可从TKE和ε计算）
- ❌ 维度过高可能导致过拟合
- ❌ 计算和存储开销增加

**适用场景**：
- 消融实验：研究哪些参数最重要
- 参数敏感性分析
- 学术研究

**数据准备**：
```python
# 伪代码
condition = np.array([
    sample['tke'],
    sample['epsilon'],
    sample['alpha'],
    sample['kappa_e'],
    sample['u_prime'],
    sample['Re_L']
])  # shape: (6,)
```

---

## 📋 参数统计范围（基于1000个样本）

### TKE统计
```
最小值：0.046875 m²/s²
最大值：0.1875 m²/s²
范围：4倍
分布：均匀采样10个点
```

### Epsilon统计
```
最小值：0.2701 m²/s³
最大值：1.0804 m²/s³
范围：4倍
分布：均匀采样10个点
```

### Alpha统计（估计范围）
```
典型范围：1.2 ~ 1.8
依赖于：TKE, epsilon, 调谐方程解
```

### Kappa_e统计（估计范围）
```
典型范围：30 ~ 60 m⁻¹
依赖于：epsilon, ν, 调谐方程解
```

### U_prime统计（估计范围）
```
最小值：≈0.177 m/s
最大值：≈0.354 m/s
计算：√(2·TKE/3)
```

### Re_L统计（估计范围）
```
最小值：≈8.1
最大值：≈130
计算：TKE²/(ε·ν)
```

---

## 🔄 参数组合结构

### 参数ID映射
```
param_id = tke_index * N_EPS_SAMPLES + eps_index
         = tke_index * 10 + eps_index
         
范围: 0 ~ 99 (共100组)
```

### 样本ID映射
```
sample_id = param_id * N_SEEDS_PER_PARAM + seed_offset
          = param_id * 10 + seed_offset

范围: 0 ~ 999 (共1000个样本)
```

### 种子计算
```
seed = param_id * N_SEEDS_PER_PARAM + seed_offset
     = param_id * 10 + seed_offset

范围: 0 ~ 999
```

---

## 🎲 关于随机种子的说明

### ⚠️ 重要：seed不应作为条件输入！

**原因**：
1. **无物理意义**：seed只是一个随机数，不代表任何物理量
2. **已体现在噪声中**：seed的作用已经通过对应的白噪声场z(x)体现了
3. **学习目标错误**：FNO应该学习"噪声→湍流"的映射，而不是"数字→湍流"

**正确做法**：
- seed用于生成对应的白噪声场z(x)
- z(x)作为FNO的输入（与c并行输入）
- FNO学习映射：(z(x), c) → u'(x)

---

## 📊 数据文件结构

### 湍流场数据
```
TurbulenceData_1000_Tuned/
├── velocity_fields/
│   ├── field_0000.npy  [256×256×2]
│   ├── field_0001.npy
│   └── ...
├── metadata/
│   ├── generation_metadata.json
│   └── training_config.json
└── spectra_plots/
    └── spectrum_*.png
```

### 白噪声场数据
```
WhiteNoiseData_1000/
├── noise_fields/
│   ├── noise_0000.npy  [256×256]
│   ├── noise_0001.npy
│   └── ...
└── metadata/
    ├── noise_metadata.json
    └── noise_turbulence_mapping.json
```

### 对应关系
```
noise_XXXX.npy ↔ field_XXXX.npy
(相同的XXXX表示配对样本，使用相同的seed)
```

---

## 🚀 实施建议

### 阶段1：起步阶段（推荐方案A）
1. 使用最小条件输入 `c = [TKE, epsilon]`
2. 验证FNO基本架构
3. 快速迭代和调试

### 阶段2：优化阶段（推荐方案B）⭐
1. 切换到扩展条件输入 `c = [TKE, epsilon, alpha, kappa_e]`
2. 提升生成质量
3. 加速训练收敛
4. **这是推荐的最终方案**

### 阶段3：实验阶段（可选方案C）
1. 完整条件输入做消融实验
2. 分析各参数的重要性
3. 为论文准备实验数据

---

## 📝 条件归一化建议

### 归一化方法

**方法1：标准化（推荐）**
```python
c_normalized = (c - c_mean) / c_std
```

**方法2：Min-Max归一化**
```python
c_normalized = (c - c_min) / (c_max - c_min)
```

**方法3：对数归一化（如果跨度大）**
```python
c_normalized = (log(c) - log(c_mean)) / log_std
```

### 各参数归一化参数（示例）

| 参数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
| TKE | 0.1172 | 0.0415 | 0.0469 | 0.1875 |
| epsilon | 0.6752 | 0.2401 | 0.2701 | 1.0804 |
| alpha | ~1.5 | ~0.2 | ~1.2 | ~1.8 |
| kappa_e | ~45 | ~10 | ~30 | ~60 |

**注意**：以上统计值需要从实际生成的metadata中提取！

---

## 🔧 代码实现示例

### 方案A代码框架
```python
# 读取条件
tke = metadata['tke']
epsilon = metadata['epsilon']
condition = np.array([tke, epsilon], dtype=np.float32)

# 归一化
condition = (condition - condition_mean) / condition_std

# 读取噪声
noise = np.load('noise_XXXX.npy')

# FNO输入
# 方式1: 拼接
input_tensor = np.concatenate([noise[..., None], 
                               condition * np.ones_like(noise)[..., None]], axis=-1)

# 方式2: 通过条件嵌入网络
condition_embedding = condition_encoder(condition)
```

### 方案B代码框架
```python
# 读取条件（包含衍生参数）
tke = metadata['tke']
epsilon = metadata['epsilon']
alpha = metadata['alpha']
kappa_e = metadata['kappa_e']
condition = np.array([tke, epsilon, alpha, kappa_e], dtype=np.float32)

# 其他步骤同方案A
```

---

## ✅ 总结

### 推荐方案对比

| 维度 | 方案A | **方案B** ⭐ | 方案C |
|------|-------|------------|-------|
| 条件维度 | 2 | **4** | 6 |
| 复杂度 | 低 | **中** | 高 |
| 性能预期 | 中 | **高** | 高 |
| 训练速度 | 快 | **中等** | 慢 |
| 推荐指数 | ⭐⭐⭐ | **⭐⭐⭐⭐⭐** | ⭐⭐ |

### 最终建议

**推荐使用方案B作为主要方案**：
```python
c = [TKE, epsilon, alpha, kappa_e]  # 4维条件输入
```

**理由**：
1. 物理意义清晰
2. 包含关键参数
3. 性能和复杂度平衡最优
4. 便于后续分析和解释

---

## 📚 参考信息

### 元数据文件位置
- 湍流场元数据：`TurbulenceData_1000_Tuned/metadata/generation_metadata.json`
- 白噪声元数据：`WhiteNoiseData_1000/metadata/noise_metadata.json`
- 映射关系：`WhiteNoiseData_1000/metadata/noise_turbulence_mapping.json`

### 能谱模型
- Von Kármán-Pao调谐模型
- 文献：基于调谐方程的能谱生成方法

---

**文档版本**：v1.0  
**创建日期**：2025-11-06  
**作者**：Claude  
**用途**：FNO湍流生成模型训练 - 条件输入设计
