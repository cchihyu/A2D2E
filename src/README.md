# src — Shared Python Library

This directory contains the core library used by all Python experiments.

## Files

### `algorithms.py`
Implements all five main-effect estimators. Each returns `(grid, centred_effect)`.

| Function | Method | Notes |
|----------|--------|-------|
| `pd_curve` | Partial Dependence | Marginalises over full training set |
| `m_curve` | M-plot | Conditions on local neighbourhood |
| `ale_curve` | Accumulated Local Effects | Quantile bins, finite differences |
| `dale_curve` | Differential ALE | Requires a gradient function |
| `a2d2e_curve` | **A2D2E (ours)** | D-optimal 2^D designs per bin, closed-form slope |

Key internals: `_bin_boundaries`, `_bin_members`, `_bin_index`, `_d_optimal_design`, `_estimate_beta`.

### `benchmarks.py`
Defines seven additive benchmark environments (`f0`–`f6`), accessed via the `ENVS` dict.

| Function | Description |
|----------|-------------|
| `sample(env, n, dependence, noise_frac, seed)` | Draw training data with optional feature dependence |
| `true_effect_on_grid(env, d, grid)` | Evaluate the ground-truth main effect for dimension `d` |

Dependence levels: `independent`, `low`, `high`, `veryhigh` (Gaussian copula).

### `models.py`
Model fitting and gradient utilities.

| Function | Description |
|----------|-------------|
| `load_params(path)` | Load a `PARAMS` dict from a `.py` param file |
| `fit_knn` | k-nearest neighbours regressor (sklearn) |
| `fit_nn` | Multi-layer perceptron with standardisation pipeline (sklearn) |
| `fit_rf` | Random forest regressor (sklearn) |
| `fit_gp` | Gaussian process regressor with RBF kernel (sklearn) |
| `get_gradient` | Finite-difference gradient for any `f_hat` |
| `get_nn_gradient` | Exact gradient via sklearn MLP weights |
| `get_gp_gradient` | Analytic gradient for the fitted GP |

## Usage

Add the repository root to your Python path (or use the provided `pyrightconfig.json`), then:

```python
import sys
sys.path.insert(0, "src")

from algorithms import a2d2e_curve
from benchmarks import ENVS, sample, true_effect_on_grid
from models import load_params, fit_knn
```
