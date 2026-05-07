"""
Appendix C: Wall-clock complexity tables.

Self-contained script (no dependency on src/).
Produces LaTeX Tables comparing PD / ALE / A2D2E wall-clock time.

python experiments/appendix_c/run.py --outfile complexity.tex
"""

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# ── Data ──────────────────────────────────────────────────────────────────────

def additive_function(X):
    """f(x_1,...,x_p) = x_1 + x_2^2. Remaining dims have zero effect."""
    X = np.asarray(X)
    y = X[:, 0].copy()
    if X.shape[1] >= 2:
        y += X[:, 1] ** 2
    return y


def make_data(n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, p))
    y = additive_function(X)
    return X, y


def fit_1nn(X, y):
    model = KNeighborsRegressor(n_neighbors=1, algorithm="auto", n_jobs=1)
    model.fit(X, y)
    return model


# ── Utilities ─────────────────────────────────────────────────────────────────

def bin_indices(x, K, lower=0.0, upper=1.0):
    edges = np.linspace(lower, upper, K + 1)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, K - 1)
    return idx, edges


def wall_time_seconds(func):
    gc.collect()
    t0 = time.perf_counter()
    _ = func()
    return time.perf_counter() - t0


def mean_se(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


# ── PD ────────────────────────────────────────────────────────────────────────

def pd_effect_single(model, X, d, K):
    n, p = X.shape
    grid = (np.arange(K) + 0.5) / K
    Xq = np.repeat(X, K, axis=0)
    Xq[:, d] = np.tile(grid, n)
    pred = model.predict(Xq).reshape(n, K)
    effect = pred.mean(axis=0)
    return grid, effect - effect.mean()


def pd_effect_all(model, X, K):
    return [pd_effect_single(model, X, d, K) for d in range(X.shape[1])]


# ── ALE ───────────────────────────────────────────────────────────────────────

def ale_effect_single(model, X, d, K):
    idx, edges = bin_indices(X[:, d], K)
    left = edges[idx]
    right = edges[idx + 1]
    X_left = X.copy(); X_left[:, d] = left
    X_right = X.copy(); X_right[:, d] = right
    diff = model.predict(X_right) - model.predict(X_left)
    counts = np.bincount(idx, minlength=K)
    sums = np.bincount(idx, weights=diff, minlength=K)
    bin_mean = np.divide(sums, counts, out=np.zeros(K), where=counts > 0)
    effect = np.cumsum(bin_mean)
    effect = effect - np.average(effect, weights=np.maximum(counts, 1e-12))
    return 0.5 * (edges[:-1] + edges[1:]), effect


def ale_effect_all(model, X, K):
    return [ale_effect_single(model, X, d, K) for d in range(X.shape[1])]


# ── A2D2E ─────────────────────────────────────────────────────────────────────

def hypercube_signs(p):
    m = 2 ** p
    codes = ((np.arange(m)[:, None] >> np.arange(p)) & 1)
    return 2.0 * codes - 1.0


def a2d2e_local_betas(model, X, delta, clip_domain=None, batch_size=None):
    """Estimate full local coefficient vector beta_i for every observation."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    signs = hypercube_signs(p)
    n_vertices = signs.shape[0]
    half_width = delta / 2.0
    betas = np.empty((n, p), dtype=float)

    if batch_size is None:
        batch_size = max(1, int(200_000 // n_vertices))

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end]
        X_vertices = X_batch[:, None, :] + half_width * signs[None, :, :]
        if clip_domain is not None:
            lo, hi = clip_domain
            X_vertices = np.clip(X_vertices, lo, hi)
        Xq = X_vertices.reshape(-1, p)
        yq = model.predict(Xq).reshape(end - start, n_vertices)
        betas[start:end] = (yq @ signs) / (n_vertices * half_width)

    return betas


def a2d2e_effect_from_betas(X, betas, d, K):
    idx, edges = bin_indices(X[:, d], K)
    counts = np.bincount(idx, minlength=K)
    sums = np.bincount(idx, weights=betas[:, d], minlength=K)
    bin_slopes = np.divide(sums, counts, out=np.zeros(K), where=counts > 0)
    widths = np.diff(edges)
    effect = np.cumsum(widths * bin_slopes)
    effect = effect - np.average(effect, weights=np.maximum(counts, 1e-12))
    return 0.5 * (edges[:-1] + edges[1:]), effect


def a2d2e_effect_single(model, X, d, K, delta, clip_domain=None):
    betas = a2d2e_local_betas(model, X, delta=delta, clip_domain=clip_domain)
    return a2d2e_effect_from_betas(X, betas, d, K)


def a2d2e_effect_all(model, X, K, delta, clip_domain=None):
    betas = a2d2e_local_betas(model, X, delta=delta, clip_domain=clip_domain)
    return [a2d2e_effect_from_betas(X, betas, d, K) for d in range(X.shape[1])]


# ── Benchmarks ────────────────────────────────────────────────────────────────

def benchmark_single_dim(ns=(250, 500, 1000, 2000, 4000), p=4, K=40, n_runs=5,
                          delta=None, clip_domain=None):
    if delta is None:
        delta = 1.0 / K

    rows = []
    for n in ns:
        times = {"PD": [], "ALE": [], "A2D2E": []}
        for seed in range(n_runs):
            X, y = make_data(n=n, p=p, seed=seed)
            model = fit_1nn(X, y)
            _ = model.predict(X[:2])  # warm-up
            times["PD"].append(wall_time_seconds(lambda: pd_effect_single(model, X, d=0, K=K)))
            times["ALE"].append(wall_time_seconds(lambda: ale_effect_single(model, X, d=0, K=K)))
            times["A2D2E"].append(wall_time_seconds(
                lambda: a2d2e_effect_single(model, X, d=0, K=K, delta=delta, clip_domain=clip_domain)
            ))

        pd_mean, pd_se = mean_se(times["PD"])
        ale_mean, ale_se = mean_se(times["ALE"])
        a2d2e_mean, a2d2e_se = mean_se(times["A2D2E"])
        rows.append({"n": n, "PD_mean": pd_mean, "PD_se": pd_se,
                     "ALE_mean": ale_mean, "ALE_se": ale_se,
                     "A2D2E_mean": a2d2e_mean, "A2D2E_se": a2d2e_se})

    return pd.DataFrame(rows)


def benchmark_all_dims(ps=(2, 3, 4, 5, 8), n=500, K=40, n_runs=5,
                        delta=None, clip_domain=None):
    if delta is None:
        delta = 1.0 / K

    rows = []
    for p in ps:
        times = {"PD": [], "ALE": [], "A2D2E": []}
        for seed in range(n_runs):
            X, y = make_data(n=n, p=p, seed=10_000 + seed)
            model = fit_1nn(X, y)
            _ = model.predict(X[:2])  # warm-up
            times["PD"].append(wall_time_seconds(lambda: pd_effect_all(model, X, K=K)))
            times["ALE"].append(wall_time_seconds(lambda: ale_effect_all(model, X, K=K)))
            times["A2D2E"].append(wall_time_seconds(
                lambda: a2d2e_effect_all(model, X, K=K, delta=delta, clip_domain=clip_domain)
            ))

        pd_mean, pd_se = mean_se(times["PD"])
        ale_mean, ale_se = mean_se(times["ALE"])
        a2d2e_mean, a2d2e_se = mean_se(times["A2D2E"])
        rows.append({"p": p, "PD_mean": pd_mean, "PD_se": pd_se,
                     "ALE_mean": ale_mean, "ALE_se": ale_se,
                     "A2D2E_mean": a2d2e_mean, "A2D2E_se": a2d2e_se})

    return pd.DataFrame(rows)


# ── LaTeX ─────────────────────────────────────────────────────────────────────

def latex_complexity_table(single_df, all_df, digits=3):
    def get(df, i, col):
        return df.iloc[i][col] if i < len(df) else ""

    def fmt_int(x):
        return "" if x == "" else str(int(x))

    def fmt_mean_se(mean, se):
        if mean == "" or se == "":
            return ""
        return rf"${mean:.{digits}f} \pm {se:.{digits}f}$"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Mean wall-clock time (seconds) for PD, ALE, and A2D2E. "
        r"Values: mean $\pm$ Monte Carlo SE over repeated runs. "
        r"\textit{Left}: single variable ($p=4$, varying $n$). "
        r"\textit{Right}: all variables ($n=500$, varying $p$). "
        r"A2D2E estimates the full local coefficient vector once and reuses "
        r"coordinates across dimensions.}",
        r"\small",
        r"\begin{tabular}{|c|ccc|c|ccc|}",
        r"\hline",
        r"\multicolumn{4}{|c|}{\textbf{Single variable: fixed $p=4$, varying $n$}} & "
        r"\multicolumn{4}{c|}{\textbf{All variables: fixed $n=500$, varying $p$}} \\",
        r"\hline",
        r"$n$ & PD (s) & ALE (s) & A2D2E (s) & $p$ & PD (s) & ALE (s) & A2D2E (s) \\",
        r"\hline",
    ]

    for i in range(max(len(single_df), len(all_df))):
        row = [
            fmt_int(get(single_df, i, "n")),
            fmt_mean_se(get(single_df, i, "PD_mean"), get(single_df, i, "PD_se")),
            fmt_mean_se(get(single_df, i, "ALE_mean"), get(single_df, i, "ALE_se")),
            fmt_mean_se(get(single_df, i, "A2D2E_mean"), get(single_df, i, "A2D2E_se")),
            fmt_int(get(all_df, i, "p")),
            fmt_mean_se(get(all_df, i, "PD_mean"), get(all_df, i, "PD_se")),
            fmt_mean_se(get(all_df, i, "ALE_mean"), get(all_df, i, "ALE_se")),
            fmt_mean_se(get(all_df, i, "A2D2E_mean"), get(all_df, i, "A2D2E_se")),
        ]
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\label{tab:complexity}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", type=str, default="complexity.tex")
    parser.add_argument("--n-runs",  type=int, default=5)
    parser.add_argument("--digits",  type=int, default=3)
    args = parser.parse_args()

    K = 40
    delta = 1.0 / K

    print("Running single-dim benchmark...")
    single_df = benchmark_single_dim(
        ns=(250, 500, 1000, 2000, 4000),
        p=4, K=K, n_runs=args.n_runs, delta=delta,
    )

    print("Running all-dims benchmark...")
    all_df = benchmark_all_dims(
        ps=(2, 3, 4, 5, 8),
        n=2000, K=K, n_runs=args.n_runs, delta=delta,
    )

    print(single_df.to_string())
    print(all_df.to_string())

    latex_code = latex_complexity_table(single_df, all_df, digits=args.digits)
    with open(args.outfile, "w") as f:
        f.write(latex_code + "\n")
    print(f"\nSaved LaTeX to: {args.outfile}")


if __name__ == "__main__":
    main()
