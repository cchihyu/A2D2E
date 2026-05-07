"""
Figure 3: GPR length-scale sensitivity study.

Self-contained script (no dependency on src/).

python experiments/figure3/run.py --ls_idx 0 --rep 0 --out_dir results_gp_sensitivity

Each job: one (ls, rep) pair, all 5 methods evaluated.
Results saved as results_spike/ls{ii}_rep{rrr}.npz
"""

import argparse
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

LS_VALUES = [0.001, 0.003, 0.005, 0.007, 0.009,
             0.01, 0.03, 0.05, 0.07, 0.09,
             0.1, 0.3, 0.5, 0.7, 0.9]


def true_f(X):
    return np.exp(-100 * (X[:, 0] - 0.5) ** 2) + X[:, 1]


def true_main_effect(d, grid):
    if d == 0:
        return np.exp(-100 * (grid - 0.5) ** 2) - np.mean(np.exp(-100 * (grid - 0.5) ** 2))
    elif d == 1:
        return grid - 0.5
    else:
        return np.zeros_like(grid)


def sample_data(n=200, noise_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = np.clip(x1 + rng.normal(0, 0.05, n), 0, 1)
    x3 = np.clip(x1 + rng.normal(0, 0.05, n), 0, 1)
    X = np.column_stack([x1, x2, x3])
    y_clean = true_f(X)
    sig2 = noise_frac * np.var(y_clean)
    y = y_clean + rng.normal(0, np.sqrt(sig2), n)
    return X, y


def fit_gp_fixed_ls(X, y, length_scale):
    scaler_X = StandardScaler()
    Xs = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    ys = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = (ConstantKernel(1.0, constant_value_bounds="fixed") *
              RBF(length_scale=length_scale, length_scale_bounds="fixed") +
              WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)))

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=False)
    gp.fit(Xs, ys)

    def f_hat(Xq):
        Xqs = scaler_X.transform(Xq)
        return scaler_y.inverse_transform(gp.predict(Xqs).reshape(-1, 1)).ravel()

    def grad_fn(Xq, d):
        h = 1e-4
        Xp = Xq.copy(); Xp[:, d] += h
        Xm = Xq.copy(); Xm[:, d] -= h
        return (f_hat(Xp) - f_hat(Xm)) / (2 * h)

    return f_hat, grad_fn


N_GRID = 100
K_BINS = 40


def make_grid(X, d):
    return np.linspace(X[:, d].min(), X[:, d].max(), N_GRID)


def pd_curve(f_hat, X, d):
    grid = make_grid(X, d)
    effects = np.array([
        f_hat(np.column_stack([
            X[:, i] if i != d else np.full(len(X), g)
            for i in range(X.shape[1])
        ])).mean() for g in grid
    ])
    return grid, effects - effects.mean()


def m_curve(f_hat, X, d):
    grid = make_grid(X, d)
    bins = np.array_split(np.argsort(X[:, d]), K_BINS)
    bin_means = np.array([X[b, d].mean() for b in bins if len(b) > 0])
    bin_effects = np.array([f_hat(X[b]).mean() for b in bins if len(b) > 0])
    effects = np.interp(grid, bin_means, bin_effects)
    return grid, effects - effects.mean()


def ale_curve(f_hat, X, d):
    quantiles = np.unique(np.percentile(X[:, d], np.linspace(0, 100, K_BINS + 1)))
    if len(quantiles) < 2:
        return make_grid(X, d), np.zeros(N_GRID)
    ale = np.zeros(len(quantiles) - 1)
    for k in range(len(quantiles) - 1):
        idx = (X[:, d] >= quantiles[k]) & (X[:, d] <= quantiles[k + 1])
        if idx.sum() == 0:
            continue
        Xr = X[idx].copy(); Xr[:, d] = quantiles[k + 1]
        Xl = X[idx].copy(); Xl[:, d] = quantiles[k]
        ale[k] = (f_hat(Xr) - f_hat(Xl)).mean()
    cum_ale = np.cumsum(ale) - np.cumsum(ale).mean()
    mid_points = 0.5 * (quantiles[:-1] + quantiles[1:])
    grid = make_grid(X, d)
    return grid, np.interp(grid, mid_points, cum_ale) - np.interp(grid, mid_points, cum_ale).mean()


def dale_curve(f_hat, grad_fn, X, d):
    quantiles = np.unique(np.percentile(X[:, d], np.linspace(0, 100, K_BINS + 1)))
    if len(quantiles) < 2:
        return make_grid(X, d), np.zeros(N_GRID)
    ale = np.zeros(len(quantiles) - 1)
    for k in range(len(quantiles) - 1):
        idx = (X[:, d] >= quantiles[k]) & (X[:, d] <= quantiles[k + 1])
        if idx.sum() == 0:
            continue
        ale[k] = grad_fn(X[idx], d).mean() * (quantiles[k + 1] - quantiles[k])
    cum_ale = np.cumsum(ale) - np.cumsum(ale).mean()
    mid_points = 0.5 * (quantiles[:-1] + quantiles[1:])
    grid = make_grid(X, d)
    return grid, np.interp(grid, mid_points, cum_ale) - np.interp(grid, mid_points, cum_ale).mean()


def a2d2e_curve(f_hat, X, d, delta=0.025):
    quantiles = np.unique(np.percentile(X[:, d], np.linspace(0, 100, K_BINS + 1)))
    if len(quantiles) < 2:
        return make_grid(X, d), np.zeros(N_GRID)
    ale = np.zeros(len(quantiles) - 1)
    for k in range(len(quantiles) - 1):
        idx = np.where((X[:, d] >= quantiles[k]) & (X[:, d] <= quantiles[k + 1]))[0]
        if len(idx) == 0:
            continue
        z = 0.5 * (quantiles[k] + quantiles[k + 1])
        diffs = []
        for i in idx:
            xi = X[i].copy()
            xr = xi.copy(); xr[d] = min(z + delta, 1.0)
            xl = xi.copy(); xl[d] = max(z - delta, 0.0)
            width = xr[d] - xl[d]
            if width > 0:
                diffs.append(
                    (f_hat(xr.reshape(1, -1)) - f_hat(xl.reshape(1, -1)))[0] / width
                )
        if diffs:
            ale[k] = np.mean(diffs) * (quantiles[k + 1] - quantiles[k])
    cum_ale = np.cumsum(ale) - np.cumsum(ale).mean()
    mid_points = 0.5 * (quantiles[:-1] + quantiles[1:])
    grid = make_grid(X, d)
    return grid, np.interp(grid, mid_points, cum_ale) - np.interp(grid, mid_points, cum_ale).mean()


def compute_ormse(method_fn, X, D=3):
    rmses = []
    for d in range(D):
        grid, est = method_fn(d)
        true = true_main_effect(d, grid)
        rmses.append(np.sqrt(np.mean((est - true) ** 2)))
    return float(np.mean(rmses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ls_idx",  type=int, required=True)
    parser.add_argument("--rep",     type=int, required=True)
    parser.add_argument("--out_dir", type=str, default="results_gp_sensitivity")
    args = parser.parse_args()

    ls = LS_VALUES[args.ls_idx]
    X, y = sample_data(n=200, noise_frac=0.3, seed=args.rep)
    f_hat, grad_fn = fit_gp_fixed_ls(X, y, length_scale=ls)

    gp_mse = float(np.mean((f_hat(X) - true_f(X)) ** 2))

    method_fns = {
        "PD":    lambda d: pd_curve(f_hat, X, d),
        "M":     lambda d: m_curve(f_hat, X, d),
        "ALE":   lambda d: ale_curve(f_hat, X, d),
        "DALE":  lambda d: dale_curve(f_hat, grad_fn, X, d),
        "A2D2E": lambda d: a2d2e_curve(f_hat, X, d),
    }

    ormses = {name: compute_ormse(fn, X, D=3) for name, fn in method_fns.items()}

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"ls{args.ls_idx:02d}_rep{args.rep:03d}.npz")
    np.savez(out_file, gp_mse=gp_mse, ls=ls, ls_idx=args.ls_idx, rep=args.rep,
             **{f"ormse_{m}": ormses[m] for m in ormses})
    print(f"Saved {out_file} | ls={ls} | " +
          " | ".join(f"{m}={ormses[m]:.4f}" for m in ormses))


if __name__ == "__main__":
    main()
