"""
Main ORMSE comparison experiment (Table 2).

python experiments/main_comparison/run.py \\
    --env f0 --method a2d2e --model knn \\
    --dependence independent --n-train 300 --n-reps 100 \\
    --noise-frac 0.1 --K 40 --delta 0.025 \\
    --param-dir model_params --outdir results --verbose
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks import ENVS, sample, true_effect_on_grid
from algorithms import pd_curve, m_curve, ale_curve, a2d2e_curve, dale_curve
from models import load_params, fit_knn, get_gradient


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def rmse(a, b):
    return float(np.sqrt(mse(a, b)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        type=str,   required=True, choices=sorted(ENVS.keys()))
    parser.add_argument("--method",     type=str,   required=True,
                        choices=["pd", "m", "ale", "a2d2e", "dale"])
    parser.add_argument("--model",      type=str,   required=True,
                        choices=["knn"])
    parser.add_argument("--dependence", type=str,   default="independent",
                        choices=["independent", "low", "high", "veryhigh"])
    parser.add_argument("--n-train",    type=int,   required=True)
    parser.add_argument("--n-reps",     type=int,   required=True)
    parser.add_argument("--noise-frac", type=float, default=0.1)
    parser.add_argument("--n-grid",     type=int,   default=100)
    parser.add_argument("--K",          type=int,   default=40)
    parser.add_argument("--delta",      type=float, default=0.025)
    parser.add_argument("--eps",        type=float, default=1e-4)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--param-dir",  type=str,   default="model_params")
    parser.add_argument("--outdir",     type=str,   default="results")
    parser.add_argument("--save-curves", action="store_true")
    parser.add_argument("--verbose",    action="store_true")
    return parser.parse_args()


def fit_model(model_name, X, y, params, verbose=False):
    if model_name == "knn":
        return None, fit_knn(X, y, params=params, verbose=verbose)
    raise ValueError(model_name)


def build_grad_fn(method, f_hat, d, eps):
    if method != "dale":
        return None
    return lambda Xq: get_gradient(f_hat, Xq, d, eps=eps)


def run_method(method, f_hat, X, d, n_grid, K, delta, eps, grad_fn=None):
    if method == "pd":
        return pd_curve(f_hat=f_hat, X_train=X, d=d, n_grid=n_grid)
    if method == "m":
        return m_curve(f_hat=f_hat, X_train=X, d=d, n_grid=n_grid, K=K)
    if method == "ale":
        return ale_curve(f_hat=f_hat, X_train=X, d=d, n_grid=n_grid, K=K)
    if method == "a2d2e":
        return a2d2e_curve(f_hat=f_hat, X_train=X, d=d, n_grid=n_grid, K=K, delta=delta)
    if method == "dale":
        return dale_curve(f_hat=f_hat, X_train=X, d=d, n_grid=n_grid, K=K,
                          grad_fn=grad_fn, eps=eps)
    raise ValueError(method)


def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    env = ENVS[args.env]
    D = env["D"]

    param_file = os.path.join(args.param_dir, f"{args.env}_{args.model}.py")
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    params = load_params(param_file)

    run_name = (
        f"env-{args.env}_method-{args.method}_model-{args.model}"
        f"_dep-{args.dependence}_n-{args.n_train}_reps-{args.n_reps}"
        f"_noise-{args.noise_frac}_seed-{args.seed}"
    )
    out_path = os.path.join(args.outdir, run_name + ".json")

    results = {
        "config":          vars(args).copy(),
        "param_file":      param_file,
        "env_name":        env["name"],
        "env_description": env["description"],
        "D":               D,
        "replications":    [],
    }

    t0 = time.time()

    for rep in range(args.n_reps):
        rep_seed = args.seed + rep

        X, y = sample(env=env, n=args.n_train, dependence=args.dependence,
                      noise_frac=args.noise_frac, seed=rep_seed)

        _, f_hat = fit_model(args.model, X, y, params, verbose=args.verbose)
        train_rmse = rmse(f_hat(X), y)

        per_dim = {}
        rmse_per_dim = []

        for d in range(D):
            grad_fn = build_grad_fn(args.method, f_hat, d, args.eps)

            grid, est_effect = run_method(
                method=args.method, f_hat=f_hat, X=X, d=d,
                n_grid=args.n_grid, K=args.K, delta=args.delta,
                eps=args.eps, grad_fn=grad_fn,
            )
            true_effect = true_effect_on_grid(env, d, grid)

            effect_mse = mse(est_effect, true_effect)
            effect_rmse = rmse(est_effect, true_effect)
            rmse_per_dim.append(effect_rmse)

            dim_result = {"effect_mse": effect_mse, "effect_rmse": effect_rmse}
            if args.save_curves:
                dim_result["grid"] = np.asarray(grid).tolist()
                dim_result["est_effect"] = np.asarray(est_effect).tolist()
                dim_result["true_effect"] = np.asarray(true_effect).tolist()

            per_dim[str(d)] = dim_result

        ormse = float(np.mean(rmse_per_dim))
        results["replications"].append({
            "rep": rep, "seed": rep_seed,
            "train_rmse": train_rmse, "ormse": ormse, "per_dim": per_dim,
        })

        if args.verbose:
            print(f"[{rep+1}/{args.n_reps}] seed={rep_seed} "
                  f"train_RMSE={train_rmse:.4f}  ORMSE={ormse:.6f}")

    ormses = np.array([r["ormse"] for r in results["replications"]])
    train_rmses = np.array([r["train_rmse"] for r in results["replications"]])

    def mean_se(arr):
        m = float(arr.mean())
        s = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return m, s

    mean_ormse, se_ormse = mean_se(ormses)
    mean_train_rmse, se_train_rmse = mean_se(train_rmses)

    per_dim_summary = {}
    for d in range(D):
        mses = np.array([r["per_dim"][str(d)]["effect_mse"] for r in results["replications"]])
        rmses_ = np.array([r["per_dim"][str(d)]["effect_rmse"] for r in results["replications"]])
        m_mse, s_mse = mean_se(mses)
        m_rmse, s_rmse = mean_se(rmses_)
        per_dim_summary[str(d)] = {
            "mean_effect_mse": m_mse, "se_effect_mse": s_mse,
            "mean_effect_rmse": m_rmse, "se_effect_rmse": s_rmse,
        }

    results["summary"] = {
        "mean_ormse": mean_ormse, "se_ormse": se_ormse,
        "mean_train_rmse": mean_train_rmse, "se_train_rmse": se_train_rmse,
        "per_dim": per_dim_summary,
        "elapsed_seconds": float(time.time() - t0),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved -> {out_path}")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
