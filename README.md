# GP Differentiation and Integration — Notebooks

Companion notebooks for the paper:

> **Differentiation and Integration of Time Series via Gaussian Process Regression for Structural Health Monitoring Applications**

---

## Contents

| Notebook | Description |
|---|---|
| `01_Illustrative_Example_1_Differentiation_Duffing.ipynb` | GP **differentiation**: recover velocity and acceleration from noisy displacement — Duffing oscillator (Figure 3) |
| `02_Illustrative_Example_2_Integration_Lorenz.ipynb` | GP **integration**: recover displacement from noisy velocity — Lorenz attractor (Figure 4) |
| `03_Application_WindTurbine_Accel_to_Displacement.ipynb` | GP **double integration**: recover displacement from noisy acceleration — wind turbine SHM application (Figures 6–7) |

All notebooks are pre-executed and self-contained — figures and results are visible directly in the browser without running anything.

---

## Repository structure

All files are in this folder:

```
notebooks/
├── 01_Illustrative_Example_1_Differentiation_Duffing.ipynb
├── 02_Illustrative_Example_2_Integration_Lorenz.ipynb
├── 03_Application_WindTurbine_Accel_to_Displacement.ipynb
├── data_generation.py           ← ODE solvers and data loaders
├── gp_optimization.py           ← NLL optimisation (Algorithms 1 & 2)
├── KalmanFilter_functions.py    ← Kalman filter + RTS smoother
├── Matern_52_state_space.py     ← Matérn 5/2 state-space matrices
├── plotting_functions.py        ← plotting utilities
└── windturbine2D.txt            ← wind turbine simulation data
```

---

## Dependencies

```
numpy
scipy
matplotlib
```

Install with:

```bash
pip install numpy scipy matplotlib
```

---

## Method overview

The core idea is to represent the unknown time series as a **Matérn 5/2 Gaussian Process** and exploit the fact that its state-space form places the signal and all its derivatives inside a single 3-dimensional latent state:

$$\mathbf{z}(t) = \begin{bmatrix} x(t) \\ \dot{x}(t) \\ \ddot{x}(t) \end{bmatrix}$$

Observations of **any** derivative are fused via a Kalman filter, and the RTS smoother recovers the full posterior over all derivatives simultaneously — including uncertainty quantification.

- **Differentiation**: observe $x(t)$, read off $\dot{x}$ and $\ddot{x}$ from the smoothed state.
- **Integration**: observe $\dot{x}(t)$, read off $x(t)$ from the smoothed state.
- **Double integration**: observe $\ddot{x}(t)$, read off $x(t)$ from the smoothed state.
