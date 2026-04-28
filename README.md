# TalarAI — AI-Driven Hydrofoil Optimization

An end-to-end machine learning pipeline for efficient 2D hydrofoil cross-section design. TalarAI encodes complex airfoil geometries into a compact 6-parameter latent space, enabling rapid aerodynamic optimization without expensive CFD simulations.

> **Senior Design Project** · Stevens Institute of Technology · Software Engineering, Class of 2026

---

## Overview

Traditional hydrofoil design relies on iterative CFD simulations or manual trial-and-error — both slow and computationally expensive. TalarAI replaces that loop with a learned low-dimensional representation of hydrofoil shapes that can be evaluated and optimized in milliseconds.

The result: a fully automated pipeline that generates, evaluates, and optimizes 2D hydrofoil geometries while respecting real-world physical and geometric constraints. Outputs are intended to support downstream physical prototyping (3D printing and towing tank experiments).

---

## Pipeline

```
Airfoil Image (PNG)
       │
       ▼
  CNN Encoder  ──►  6-parameter latent vector
       │
       ▼
Neural Decoder  ──►  80-point surface geometry (upper + lower)
       │
       ▼
   NeuralFoil  ──►  CL, CD  (lift & drag coefficients)
       │
       ▼
NOM Optimizer  ──►  minimize CD/CL  subject to geometric constraints
```

### Components

**1. Latent Parameterization**
A CNN autoencoder (trained on 1,600+ UIUC airfoil database geometries) compresses each hydrofoil shape into a 6-dimensional latent vector. The encoder maps airfoil PNG images → latent space; the decoder reconstructs 80-point surface coordinates from latent parameters.

**2. Aerodynamic Evaluation**
Reconstructed geometries are evaluated using [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) — a physics-informed neural network surrogate for XFOIL that predicts CL and CD at a specified angle of attack and Reynolds number, typically within ~2–3% of XFOIL results in the operating range.

**3. Neural Optimization Machine (NOM)**
A gradient-based optimizer over the latent space. Objective: minimize CD/CL (equivalently, maximize lift-to-drag ratio L/D). Constraints — minimum thickness at leading edge, mid-chord, and trailing edge; maximum camber; minimum trailing edge angle — are enforced via ReLU-based penalty functions.

---

## Results

On the E61 baseline airfoil at α = 2°, Re = 150,000:

| | CL | CD | L/D |
|---|---|---|---|
| XFOIL reference (Re = 200k) | 1.045 | 0.01391 | 75.1 |
| E61 baseline via NeuralFoil | 1.128 | 0.01417 | 79.6 |
| NOM optimized | 1.056 | 0.01281 | **82.5** |

Optimization converged in 38 iterations (of 500), achieving a **+3.6% improvement in L/D** over the baseline.

---

## Installation

```bash
git clone https://github.com/Rymarmar/TalarAI-Hydrofoil-Optimization
cd TalarAI-Hydrofoil-Optimization
pip install -r requirements.txt
```

---

## Repository Structure

```
pipeline/        # Core evaluation pipeline (decoder + NeuralFoil integration)
optimization/    # NOM optimizer and objective/constraint definitions
data/            # Latent parameter datasets and baseline geometries
tools/           # Visualization and debugging utilities
experiments/     # Exploratory code not used in the final pipeline
```

---

## Scope & Assumptions

- Optimization targets **2D hydrofoil cross-sections** only; 3D effects (planform, aspect ratio, tip vortices) are out of scope.
- Angle of attack and Reynolds number are held constant within each optimization run to enable fair comparison.
- The pipeline is modular — the optimizer, evaluator, or decoder can each be swapped independently.

---

## Citation

Aerodynamic evaluation powered by **NeuralFoil** (MIT License):

```bibtex
@misc{neuralfoil,
  author       = {Peter Sharpe},
  title        = {{NeuralFoil}: An airfoil aerodynamics analysis tool using physics-informed machine learning},
  year         = {2023},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/peterdsharpe/NeuralFoil}},
}

@phdthesis{aerosandbox_phd_thesis,
  title  = {Accelerating Practical Engineering Design Optimization with Computational Graph Transformations},
  author = {Sharpe, Peter D.},
  school = {Massachusetts Institute of Technology},
  year   = {2024},
}
```

Airfoil geometry data sourced from the [UIUC Airfoil Database](https://m-selig.ae.illinois.edu/ads/coord_database.html).
