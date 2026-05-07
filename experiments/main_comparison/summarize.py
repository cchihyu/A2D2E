"""
Generate LaTeX Table 2.

python experiments/main_comparison/summarize.py --results-dir results --outfile table2.tex
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


ENVS = ["f0", "f1", "f2", "f3", "f4", "f5", "f6"]
METHODS = ["pd", "m", "ale", "dale", "a2d2e"]
MODELS = ["knn"]
DEPS = ["independent", "low", "high", "veryhigh"]
NOISES = [0.1, 0.3, 0.5]

METHOD_LABEL = {"pd": "PD", "m": "M", "ale": "ALE", "dale": "DALE", "a2d2e": "A2D2E"}
MODEL_LABEL = {"knn": "KNN", "nn": "NN", "rf": "RF", "gp": "GP"}
DEP_LABEL = {
    "independent": "Independent",
    "low": "Low dependence",
    "high": "High dependence",
    "veryhigh": "Very high dependence",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--outfile",     type=str, default="table2.tex")
    parser.add_argument("--digits",      type=int, default=4)
    return parser.parse_args()


def load_results(results_dir):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for fname in Path(results_dir).glob("*.json"):
        try:
            with open(fname) as f:
                res = json.load(f)
        except Exception as e:
            print(f"[warn] could not read {fname}: {e}")
            continue

        cfg = res.get("config", {})
        env = cfg.get("env")
        method = cfg.get("method")
        model = cfg.get("model")
        dep = cfg.get("dependence", "independent")
        noise = float(cfg.get("noise_frac", 0.0)) if cfg.get("noise_frac") is not None else None

        if any(x is None for x in [env, method, model, dep, noise]):
            print(f"[warn] missing config fields in {fname}")
            continue

        summary = res.get("summary", {})

        if "mean_ormse" in summary:
            mean = float(summary["mean_ormse"])
            se = float(summary.get("se_ormse", 0.0))
        else:
            per_dim = summary.get("per_dim", {})
            if not per_dim:
                print(f"[warn] no summary/per_dim in {fname}")
                continue
            rmses = [float(v["mean_effect_rmse"]) for v in per_dim.values()
                     if "mean_effect_rmse" in v]
            ses = [float(v.get("se_effect_rmse", 0.0)) for v in per_dim.values()
                   if "mean_effect_rmse" in v]
            if not rmses:
                continue
            mean = float(np.mean(rmses))
            se = float(np.sqrt(np.mean(np.array(ses) ** 2)))

        data[dep][noise][model][env][method] = (mean, 1.96 * se)

    return data


def fmt(mean, ci, digits, style=None):
    s = f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"
    if style == "best":
        return f"\\textbf{{{s}}}"
    elif style == "second":
        return f"\\underline{{{s}}}"
    return s


def build_table(data, dep, noise, digits):
    n_methods = len(METHODS)
    col_spec = "ll" + "c" * n_methods

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Mean ORMSE ($\pm$ 95\% CI) under \textit{"
        + DEP_LABEL[dep]
        + r"} with noise fraction "
        + f"{noise}"
        + r". Bold = best per row; underline = second best.}"
    )
    lines.append(
        f"\\label{{tab:ormse_{dep}_noise_{str(noise).replace('.', 'p')}}}"
    )
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(f"Model & Function & " + " & ".join(METHOD_LABEL[m] for m in METHODS) + r" \\")
    lines.append(r"\midrule")

    dep_block = data.get(dep, {})
    noise_block = dep_block.get(noise, {})

    for model in MODELS:
        for i, env in enumerate(ENVS):
            row_vals = {
                method: noise_block.get(model, {}).get(env, {}).get(method)
                for method in METHODS
            }
            valid = {m: v for m, v in row_vals.items() if v is not None}
            ranked = sorted(valid.keys(), key=lambda m: valid[m][0])
            best = ranked[0] if len(ranked) >= 1 else None
            second = ranked[1] if len(ranked) >= 2 else None

            cells = []
            for method in METHODS:
                entry = row_vals[method]
                if entry is None:
                    cells.append("--")
                else:
                    mean, ci = entry
                    style = "best" if method == best else ("second" if method == second else None)
                    cells.append(fmt(mean, ci, digits, style=style))

            row_prefix = (
                f"\\multirow{{{len(ENVS)}}}{{*}}{{{MODEL_LABEL[model]}}} & {env}"
                if i == 0 else f" & {env}"
            )
            lines.append(row_prefix + " & " + " & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    args = parse_args()
    data = load_results(args.results_dir)

    if not data:
        print("No results found. Check --results-dir.")
        return

    tables = []
    for dep in DEPS:
        for noise in NOISES:
            tables.append(build_table(data, dep, noise, args.digits))

    with open(args.outfile, "w") as f:
        f.write("\n\n".join(tables) + "\n")

    print(f"Saved LaTeX tables to: {args.outfile}")


if __name__ == "__main__":
    main()
