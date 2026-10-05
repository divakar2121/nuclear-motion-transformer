# Nuclear Motion Transformer

**Physics-Informed Machine Learning for Characterizing Active Brownian Motion of Cell Nuclei**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements a **three-system bridge** for analyzing nuclear motion in biological systems, combining physics simulation, machine learning classification, and neural ordinary differential equations (Neural ODEs).

The work targets the PhD project: **"Characterization of active Brownian motion of cell nuclei"** (Fischer-Friedrich Lab, TU Dresden).

## The Three Systems

### 1. **Atomic Gradient Bridge** (Physics Simulation)
**File:** `atomic_gradient_bridge.py`

Simulates 30 particles interacting via Lennard-Jones potential using gradient descent as a force computation engine.

**Key Insight:** Gradient descent computes **physical forces on atoms**, not learning signals. Atoms physically move in 3D space to minimize potential energy.

```python
# Gradient = Force on atoms
positions = nn.Parameter(torch.randn(30, 3))  # Atom positions are parameters
energy = lennard_jones(positions)              # LJ potential
energy.backward()                              # Compute forces
optimizer.step()                               # Atoms move!
```

**Results:**
- Energy: 4.3M → -97 (100% reduction)
- 100 simulation steps
- Visualizes force vectors and energy landscapes

### 2. **Physics-Informed Motion Transformer (PiMT)** (ML Classification)
**File:** `pimt_mini.py`

A transformer architecture that classifies nuclear motion states (confined, sub-diffusive, normal, active) using physics-informed features.

**Architecture:**
- 1D CNN → Transformer Encoder → Classification Head
- Physics features: α (anomalous exponent), MSD, VACF, confinement
- Multi-task: motion state classification + alpha regression

**Results:**
- 90% classification accuracy
- Alpha correlation: 0.71
- 20,613 parameters

### 3. **Neural Dynamics** (Neural ODE)
**File:** `neural_dynamics.py`

Learns the underlying dynamics function dx/dt = f(x,t) directly from real trajectory data using a neural network.

**Key Insight:** This is **genuine machine learning** - the network learns the force field implicitly from data, not from explicit physics equations.

```python
# Learn dynamics from real data
x_current = trajectory[t]
dx_dt = model(x_current)           # Neural network predicts velocity
x_next = x_current + dx_dt * dt    # Euler integration
```

**Results:**
- Trained on 30,154 real Drosophila trajectory pairs
- Mean prediction error: 0.83 μm over 19 steps
- 8,771 parameters

## The Bridge: Three Perspectives on Gradient Descent

| System | What Gradient Updates | Interpretation |
|--------|----------------------|----------------|
| **Atomic Bridge** | Atom positions | Force on atoms (physics simulation) |
| **PiMT** | Network weights | Learning signal (ML classification) |
| **Neural Dynamics** | Network weights | Learning signal (dynamics prediction) |

**Progression:** Understanding (physics) → Analyzing (ML features) → Predicting (learned dynamics)

## Installation

```bash
# Clone repository
git clone https://github.com/divakar2121/nuclear-motion-transformer.git
cd nuclear-motion-transformer

# Create conda environment
conda create -n nuclear-motion python=3.11
conda activate nuclear-motion

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Physics Simulation

```bash
python atomic_gradient_bridge.py
```

Generates:
- `atomic_gradient_motion.gif` - 3D particle animation with force vectors
- `gradient_physics_summary.png` - Energy landscape and force analysis

### 2. Motion Classification

```bash
python pimt_mini.py
```

**Note:** Requires real Drosophila tracking data. See [Data](#data) section below.

Generates:
- Classification metrics (accuracy, confusion matrix)
- Alpha prediction correlation
- Confidence analysis

### 3. Neural Dynamics

```bash
python neural_dynamics.py
```

**Note:** Requires real Drosophila tracking data.

Generates:
- `output_neural_dynamics/trajectory_predictions.png`
- `output_neural_dynamics/velocity_field.png`
- `output_neural_dynamics/error_over_time.png`

## Data

This project uses **real Drosophila embryo tracking data** from:

> **Zenodo Dataset:** [Drosophila nuclear tracking](https://zenodo.org/records/21282216)

The data is automatically downloaded and cached in `data/zenodo/` when you run `pimt_mini.py` or `neural_dynamics.py`.

**Data characteristics:**
- 1,321 CSV files (individual particle tracks)
- 2,642 trajectories total
- 3D coordinates (x, y, z) with timestamps
- ~274,444 sliding windows (50 frames each)

## Project Structure

```
nuclear-motion-transformer/
├── atomic_gradient_bridge.py    # System 1: Physics simulation
├── pimt_mini.py                  # System 2: PiMT classifier
├── neural_dynamics.py            # System 3: Neural ODE
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/                         # Auto-downloaded tracking data
│   └── zenodo/
└── output_neural_dynamics/       # Generated visualizations
```

## Scientific Context

### The Langevin Equation

Active Brownian motion of cell nuclei is governed by:

```
dx/dt = -∇V(x) + v_active + η(t)
```

Where:
- **-∇V(x)**: Confinement forces (chromatin, nuclear envelope)
- **v_active**: Active transport (molecular motors)
- **η(t)**: Thermal noise (Brownian motion)

### Motion Classification

Based on the anomalous diffusion exponent α:

| α Range | Motion Type | Physical Interpretation |
|---------|-------------|------------------------|
| α < 0.5 | Confined | Trapped in potential well |
| 0.5 ≤ α < 0.8 | Sub-diffusive | Hindered diffusion |
| 0.8 ≤ α < 1.2 | Normal | Free Brownian motion |
| α ≥ 1.2 | Active | Directed transport |

### Physics Features

**Mean Squared Displacement (MSD):**
```
MSD(τ) = ⟨|r(t+τ) - r(t)|²⟩ ~ τ^α
```

**Velocity Autocorrelation Function (VACF):**
```
VACF(τ) = ⟨v(t) · v(t+τ)⟩
```

**Confinement Ratio:**
```
CR = max_displacement / total_path_length
```

## Key Findings

### 1. Gradient Descent as Physics Engine
- Gradient descent can simulate physical systems by treating positions as learnable parameters
- Forces emerge naturally from potential energy gradients
- Bridges optimization and physics simulation

### 2. Physics-Informed Learning
- Explicit physics features (α, MSD, VACF) improve transformer performance
- Multi-task learning (classification + regression) provides richer representations
- Confidence analysis reveals model calibration issues

### 3. Data-Driven Dynamics
- Neural ODEs can learn complex dynamics without explicit physics equations
- Prediction accuracy: 0.83 μm error over 19 timesteps
- Enables discovery of underlying force fields from data

## Limitations and Honest Assessment

**System 1 (Atomic Bridge):**
- ✅ Demonstrates physics understanding
- ❌ Not machine learning (no learned representations)
- ✅ Useful for visualization and intuition

**System 2 (PiMT):**
- ✅ Genuine ML with physics-informed features
- ⚠️ Classification accuracy limited by class imbalance
- ⚠️ Confidence calibration needs improvement

**System 3 (Neural Dynamics):**
- ✅ Genuine ML - learns from real data
- ✅ Predicts future motion
- ⚠️ Prediction error accumulates over time

## Applications

This framework can be applied to:

1. **Tumor spheroid analysis** - Compare nuclear motion in EMT vs MET states
2. **Drug screening** - Quantify effects of cytoskeletal drugs on nuclear dynamics
3. **Disease diagnostics** - Identify abnormal motion patterns in cancer cells
4. **Biophysics research** - Discover new physical mechanisms from trajectory data

## Requirements

- Python 3.11+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib
- tqdm, scikit-learn

See `requirements.txt` for full list.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{nuclear_motion_transformer2026,
  author = {Divakar, Ravi Kumar},
  title = {Nuclear Motion Transformer: Physics-Informed ML for Active Brownian Motion},
  year = {2026},
  url = {https://github.com/divakar2121/nuclear-motion-transformer}
}
```

## Related Work

- **Fischer-Friedrich Lab** - [Physics of Life, TU Dresden](https://tu-dresden.de/mn/physik/forschung/ag-fischer-friedrich)
- **Dimari et al. (2025)** - "Mesenchymal-epithelial transition reduces proliferation but increases immune evasion in tumor spheroids"
- **Hosseini et al. (2020)** - "EMT-Induced Cell-Mechanical Changes Enhance Mitotic Rounding Strength"

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

**Divakar Ravi Kumar**  
DKFZ Heidelberg  
Email: divakar2121@github.com  
GitHub: [@divakar2121](https://github.com/divakar2121)

## Acknowledgments

- Drosophila tracking data: Zenodo DOI: 10.5281/zenodo.21282216
- Physics guidance: Fischer-Friedrich Lab publications
- PyTorch community for excellent documentation

---

**Built for the PhD project: "Characterization of active Brownian motion of cell nuclei"**
