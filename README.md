GP‑DiffInt
Gaussian‑Process‑based Differentiation and Integration of Time Series
Code accompanying the paper:
Differentiation and Integration of Time Series via Gaussian Process Regression for Structural Health Monitoring Applications

📘 Overview
This repository contains the Python code used to reproduce the numerical examples and figures in the paper:

Differentiation and Integration of Time Series via Gaussian Process Regression for Structural Health Monitoring Applications

The code implements:

A state‑space representation of the Matérn 5/2 Gaussian Process
A Kalman filter + Rauch–Tung–Striebel smoother for estimating displacement, velocity, and acceleration


The method works by selecting which derivative of the GP is *observed*:

| Task | `observed_derivative` |
|---|---|
| Differentiate a displacement signal → velocity, acceleration | `0` |
| Integrate a velocity signal → displacement | `1` |
| Double-integrate an acceleration signal → displacement | `2` |

Hyperparameters (length scale and output scale) are optimised by minimising the negative log-likelihood, optionally subject to physically motivated constraints.

---

## Citation

If you use this code, please cite:

> Caglio, L., Andersen, M. S., Paltorp, M., & Katsanos, E. (2026). Differentiation and integration of time series via Gaussian process regression for structural health monitoring applications. *Measurement*, 121367.


---

## Repository structure
```
Notebooks (interactive):
├── 01_Illustrative_Example_1_Differentiation_Duffing.ipynb  # Section 4 — differentiation
├── 02_Illustrative_Example_2_Integration_Lorenz.ipynb       # Section 4 — integration
├── 03_Application_WindTurbine_Accel_to_Displacement.ipynb   # Section 5 — wind turbine
Scripts (standalone):
├── Illustrative_example_1_2nd_derivative_Duffing.py         # Section 4 paper
├── Illustrative_example_2_1st_integral_Lorenz.py            # Section 4 paper
├── Application_displ_from_accel_wind_turbine.py             # Section 5 paper
Library modules:
├── gp_optimization.py             # NLL objective, optimisation routines, state extraction
├── Matern_52_state_space.py       # Matérn 5/2 state-space matrices (A, Qd, Pinf, D0, D1, D2)
├── KalmanFilter_functions.py      # Kalman filter and RTS smoother
├── plotting_functions.py          # Plotting utilities
├── data_generation.py             # Example data utilities
└── README.md
```

---

## Requirements

Python 3.10+. Dependencies:
```
numpy
scipy
matplotlib
```
