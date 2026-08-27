<div align="center">

# 🌪️ ML4ST-Turbulence-Simulation

### Machine learning turbulence simulation framework.

Synthetic turbulence generation with parameter tuning, against a white-noise baseline.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

</div>

---

**ML4ST-Turbulence-Simulation** is a framework for machine-learning-based turbulence simulation. It generates synthetic turbulence with tunable parameters and provides a white-noise baseline for comparison.

> [!NOTE]
> 中文项目：ML4ST 机器学习湍流模拟框架——合成湍流生成、参数调优、白噪声基线。

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/ML4ST-Turbulence-Simulation.git
cd ML4ST-Turbulence-Simulation

pip install -r requirements.txt

# Generate 1000 tuned turbulence samples
python examples/generate_turbulence_1000_tuned.py

# Generate 1000 white-noise baseline samples
python examples/generate_white_noise_1000.py
```

---

## Features

- **Synthetic turbulence generation** — tunable parameter space (`docs/Parameter_Space_Condition_Input_Design.md`).
- **Baseline comparison** — white-noise generator for evaluation.
- **Extensible package** — `src/ml4st` importable module.

---

## Project Structure

```
ML4ST-Turbulence-Simulation/
├── src/ml4st/                  # importable package
├── examples/
│   ├── generate_turbulence_1000_tuned.py
│   └── generate_white_noise_1000.py
├── docs/
│   ├── MODIFICATION_GUIDE.md
│   └── Parameter_Space_Condition_Input_Design.md
└── requirements.txt
```

---

## License

MIT — free to use, modify and distribute.
