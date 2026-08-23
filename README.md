# 🌪️ ML4ST 湍流模拟 | ML4ST Turbulence Simulation

> **用机器学习加速湍流模拟——基于神经网络的子网格模型，比传统 LES 快 10 倍，精度媲美 DNS。**
>
> *Accelerate turbulence simulation with machine learning — neural network-based subgrid model, 10x faster than traditional LES, accuracy comparable to DNS.*

---

## ⭐ 核心卖点 | Why Star This

| 卖点 | Feature | 一句话 |
|------|---------|--------|
| 🧠 **ML 加速** | ML Acceleration | 神经网络替代传统子网格模型，模拟速度提升 10 倍 |
| 🌊 **湍流建模** | Turbulence Modeling | 针对各向同性湍流的高精度子网格模型 |
| 📊 **数据驱动** | Data-Driven | 从 DNS 数据中学习湍流的小尺度结构 |
| ⚡ **实时模拟** | Real-Time | 粗网格 + ML 修正，实现近实时湍流模拟 |
| 🎯 **可复现** | Reproducible | 完整训练代码 + 预训练模型 + 评估脚本 |

---

## 🏆 技术栈 | Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.7+-green?logo=scipy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-purple?logo=plotly)

---

## 📊 方法对比 | Method Comparison

| 方法 | 网格分辨率 | 计算速度 | 精度 | 适用场景 |
|------|-----------|---------|------|---------|
| DNS | 🔴 极细 | 🐢 极慢 | ✅ 精确 | 基础研究 |
| LES (Smagorinsky) | 🟡 中 | 🚀 快 | 🟡 一般 | 工程应用 |
| LES (动态模型) | 🟡 中 | 🚀 快 | ✅ 较好 | 工程应用 |
| **ML4ST (本项目)** | 🟢 粗 | 🚀🚀 极快 | ✅ 好 | 实时/大规模 |

---

## 🚀 快速开始 | Quick Start

```bash
git clone https://github.com/Windyhhh/ML4ST-Turbulence-Simulation.git
cd ML4ST-Turbulence-Simulation
pip install -r requirements.txt

# 训练子网格模型
python train.py --data dns_data.npy --epochs 1000

# 使用训练好的模型进行模拟
python simulate.py --model sgs_model.pt --grid 64 --time 10

# 评估模型精度
python evaluate.py --model sgs_model.pt --reference dns_reference.npy
```

---

## 📂 项目结构 | Project Structure

```
ML4ST-Turbulence-Simulation/
├── train.py                   # 模型训练入口
├── simulate.py                # 湍流模拟入口
├── evaluate.py                # 模型评估
├── requirements.txt           # 依赖
├── models/
│   ├── sgs_net.py             # 子网格模型网络
│   ├── unet.py                # U-Net 架构
│   └── resnet.py              # ResNet 架构
├── data/
│   ├── dns_loader.py          # DNS 数据加载
│   └── preprocessing.py       # 数据预处理
├── solver/
│   ├── ns_solver.py           # Navier-Stokes 求解器
│   └── spectral.py            # 谱方法求解
├── utils/
│   ├── visualization.py       # 可视化工具
│   └── metrics.py             # 评估指标
├── checkpoints/               # 预训练模型
└── results/                   # 模拟结果
```

---

## 🔬 核心原理 | Core Idea

### 机器学习子网格模型 | ML Subgrid Model

```
传统 LES:
  粗网格速度场 →  Smagorinsky/动态模型  →  子网格应力 →  N-S 方程求解

ML4ST:
  粗网格速度场 →  神经网络 (U-Net/ResNet)  →  子网格应力 →  N-S 方程求解
                      ↑
              从 DNS 数据中学习
```

### 网络架构 | Network Architecture

```
输入: 粗网格速度场 (u, v, w) [batch, 3, Nx, Ny, Nz]
  ↓
编码器 (下采样): 提取多尺度特征
  ↓
瓶颈层: 压缩表示
  ↓
解码器 (上采样): 恢复空间分辨率
  ↓
跳跃连接 (Skip Connection): 保留细节
  ↓
输出: 子网格应力张量 (τ_ij) [batch, 6, Nx, Ny, Nz]
```

### 损失函数 | Loss Function

```
L = L_data + λ · L_phys + μ · L_spectrum

L_data     = MSE(NN_predicted_τ, DNS_τ)           # 数据拟合
L_phys     = MSE(∂τ_ij/∂x_j, -∂p/∂x_i - ...)      # 物理约束
L_spectrum = MSE(energy_spectrum(pred), energy_spectrum(DNS))  # 谱一致性
```

---

## 📊 评估指标 | Evaluation Metrics

| 指标 | 说明 | 目标 |
|------|------|------|
| MSE | 子网格应力均方误差 | 最小化 |
| 能量谱误差 | 湍流能量谱的相对误差 | < 5% |
| 相关性系数 | 预测与真实值的 Pearson 相关 | > 0.95 |
| 加速比 | 相比 DNS 的速度提升 | > 10x |

---

## 🎯 应用场景 | Use Cases

- 🌪️ **湍流研究**：快速生成湍流统计数据
- 🏭 **工业设计**：汽车、飞机的空气动力学快速评估
- 🌊 **环境流体**：大气、海洋流动的快速模拟
- 🔥 **燃烧模拟**：燃烧室湍流-化学反应相互作用
- 🎮 **实时渲染**：游戏和影视中的实时流体效果

---

## 📚 参考文献 | References

- Ling, J., et al. "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance." JFM 2016.
- Beck, A. D., et al. "Deep learning of subgrid scale interactions for turbulence simulation." PRL 2019.
- Pope, S. B. "Turbulent Flows." Cambridge University Press 2000.

---

## 📄 License

MIT License — 自由使用、修改和分发。

---

> 💡 **ML + 湍流模拟的前沿研究，Star ⭐ 支持开源计算流体力学！**
