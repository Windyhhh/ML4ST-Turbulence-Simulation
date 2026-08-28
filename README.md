<div align="center">

# 🌊 ML4ST-Turbulence-Simulation

### Machine learning for turbulence simulation.

Generate and process large-scale turbulence data to feed ML training — bridging CFD numerics and deep learning.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?logo=numpy&logoColor=white)](https://numpy.org/)

</div>

---

**ML4ST-Turbulence-Simulation** is a machine-learning project for **turbulence simulation** that generates and processes large-scale turbulence data to support deep-learning model training — a bridge between numerical CFD and AI-driven flow prediction.

> [!NOTE]
> 中文项目：湍流模拟机器学习（ML4ST）——大规模湍流数据生成与处理，为 AI 湍流预测提供数据支持。

---

## Features

- **Data generation** — large-scale, high-quality turbulence datasets.
- **CFD + ML integration** — numeric framework feeding ML training.
- **AI-driven prediction** — supports learned turbulence models.
- **Cost-effective** — replaces expensive commercial solvers.

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/ML4ST-Turbulence-Simulation.git
cd ML4ST-Turbulence-Simulation

pip install -r requirements.txt

python src/generate_data.py    # generate turbulence data
python src/train.py            # train the surrogate model
```

---

## Project Structure

```
ML4ST-Turbulence-Simulation/
├── src/                    # data generation, model, training
├── data/                   # generated turbulence datasets
├── configs/                # parameter-space config
└── docs/                   # modification guide, blog
```

---

## License

MIT — free to use, modify and distribute.
