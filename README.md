# TalarAI-Hydrofoil-Optimization
## Goal: AI optimization of the process of making Hydro Foils

# Installation (Implementing UI once finished)

In VS Code Terminal (make location of cloned repo accessible):
```
git clone https://github.com/Rymarmar/TalarAI-Hydrofoil-Optimization
```

Then download all required packages:
```
pip install -r requirements.txt
```

## Project Overview

TalarAI is an AI-driven hydrofoil optimization framework designed to assist in the early-stage design of 2D hydrofoil cross-sections. The goal of the project is to reduce the time and computational cost associated with traditional hydrofoil design by using machine learning models to generate, evaluate, and optimize airfoil geometries under aerodynamic and physical constraints.

Rather than relying on repeated CFD simulations or manual trial-and-error design, this project uses a learned low-dimensional representation of hydrofoil shapes. These shapes are rapidly evaluated using a physics-informed surrogate model, allowing optimization to be performed efficiently while still respecting real-world design constraints.

This project is developed as part of a senior design course and is intended to support downstream physical testing (e.g., 3D printing and towing tank experiments).

---

## Pipeline Overview

The optimization pipeline consists of the following main components:

1. **Latent Parameterization**  
   Hydrofoil geometries are represented using a small set of latent parameters (currently 6). Each parameter vector corresponds to a unique 2D hydrofoil shape.

2. **Decoder (Geometry Reconstruction)**  
   A trained neural network decoder maps the latent parameters to an 80-point hydrofoil surface representation (upper and lower surfaces).

3. **Aerodynamic Evaluation**  
   The reconstructed hydrofoil geometry is evaluated using NeuralFoil, a physics-informed neural network that predicts aerodynamic coefficients such as lift coefficient (CL) and drag coefficient (CD) at a specified angle of attack and Reynolds number.

4. **Neural Optimization Machine (NOM)**  
   Optimization is performed over the latent parameter space using a constrained optimization approach. The objective is to minimize CD/CL (equivalently maximize lift-to-drag ratio) while enforcing constraints using ReLU-based penalty functions. Constraints include valid geometry, parameter bounds, and operating conditions.

5. **Output and Analysis**  
   The pipeline outputs optimized latent parameters, reconstructed hydrofoil geometries, and corresponding aerodynamic performance metrics. These results can be visualized, compared against baseline designs, and prepared for physical prototyping.

---

## Scope and Assumptions

- The optimization focuses on **2D hydrofoil cross-sections**; 3D effects such as planform shape and aspect ratio are considered outside the scope of the current pipeline.
- Angle of attack and Reynolds number are typically held constant during optimization runs to allow fair comparison between designs.
- The system is designed to be modular, allowing components (e.g., optimizer or evaluator) to be swapped or extended in future work.

---

## Repository Structure

- `pipeline/` — Core evaluation pipeline (decoder + NeuralFoil)
- `optimization/` — Optimization logic and NOM integration
- `data/` — Latent parameter datasets and baselines
- `tools/` — Visualization and debugging utilities
- `experiments/` — Archived or exploratory code not used in the final pipeline

## Neural Foil Citation here