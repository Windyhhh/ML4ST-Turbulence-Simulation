# 🌊 ML4ST Turbulence Simulation | ML4ST 湍流模拟机器学习框架

> **Machine learning framework for turbulence simulation (ML4ST). Generate synthetic turbulence fields and white noise with tuned parameters. Complete modification guide and parameter space design documentation.**
>
> 湍流模拟机器学习框架（ML4ST）。生成带调参的合成湍流场和白噪声。完整的修改指南和参数空间设计文档。

---

## 🌟 Features | 核心特性

- **Turbulence Generation** — Synthetic turbulence field generation
- **White Noise Generation** — Baseline white noise generator
- **Parameter Tuning** — Configurable turbulence parameters
- **1000 Samples** — Generate 1000 turbulence/noise samples
- **Complete Docs** — Modification guide and parameter design docs
- **Examples** — Ready-to-run example scripts

---

## 📁 Project Structure | 项目结构

```
ML4ST-Turbulence-Simulation/
├── src/
│   └── ml4st/
│       └── __init__.py              # ML4ST package
├── examples/
│   ├── generate_turbulence_1000_tuned.py  # Tuned turbulence generation
│   └── generate_white_noise_1000.py        # White noise generation
├── docs/
│   ├── MODIFICATION_GUIDE.md        # How to modify the framework
│   ├── Parameter_Space_Condition_Input_Design.md  # Parameter design
│   └── 博客要求
├── blog.md                          # Technical blog
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start | 快速开始

```bash
pip install -r requirements.txt

# Generate 1000 tuned turbulence samples
python examples/generate_turbulence_1000_tuned.py

# Generate 1000 white noise samples (baseline)
python examples/generate_white_noise_1000.py
```

---

## 🔬 Turbulence Parameters | 湍流参数

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| **mean_velocity** | Mean flow velocity | 0-20 m/s |
| **turbulence_intensity** | Turbulence intensity level | 1-20% |
| **length_scale** | Turbulent length scale | 0.01-1 m |
| **spatial_resolution** | Grid resolution | 64-512 |
| **time_steps** | Number of time steps | 100-10000 |
| **seed** | Random seed for reproducibility | Any integer |

---

## 📊 Output Format | 输出格式

Generated turbulence fields are saved as:
- **NumPy arrays** (.npy) — Raw 3D/4D arrays
- **CSV** — Flattened statistics
- **Visualization** — Matplotlib plots (optional)

### Data Shape | 数据维度

```
turbulence_field.shape = (time_steps, height, width, channels)
```

---

## 📚 References | 参考文献

1. **Pope, S. B.** (2000). *Turbulent Flows.* Cambridge University Press.
2. **Ling, J., et al.** (2016). *Reynolds averaged turbulence modelling using deep neural networks with embedded invariance.* Journal of Fluid Mechanics.
3. **ML4ST.** (2024). *Machine Learning for Spatio-Temporal Turbulence.*

---

## 📄 License | 许可证

MIT License.

---

<div align="center">

**Built with 🌊 for turbulence simulation**

[GitHub](https://github.com/Windyhhh/ML4ST-Turbulence-Simulation)

</div>
