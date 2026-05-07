"""
Generate LaTeX tables for Appendix D.2 (non-additive benchmarks).

python experiments/appendix_d2/summary.py \\
    --results-dir results_nonadditive --outfile table_nonadditive.tex
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


ENVS = ["g1", "g2", "g3", "g4", "g5", "g6", "g7"]
METHODS = ["pd", "m", "ale", "a2d2e_goldilocks", "dale"]
MODELS = ["knn", "nn", "gp"]
RHOS = [0.7, 0.9]

METHOD_LABEL = {
    "pd": "PD", "m": "M", "ale": "ALE",
    "a2d2e_goldilocks": "A2D2E-GK", "dale": "DALE",
}
MODEL_LABEL = {"knn": "KNN", "nn": "NN", "rf": "RF", "gp": "GP"}
RHO_LABEL = {
    0.0: "Independent ($\\rho=0$)",
    0.7: "Moderate ($\\rho=0.7$)",
    0.9: "High ($\\rho=0.9$)",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results_nonadditive")
    parser.add_argument("--outfile",     type=str, default="table_nonadditive.tex")
    parser.add_argument("--digits",      type=int, default=4)
    return parser.parse_args()


def load_results(results_dir):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

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
        rho = cfg.get("rho", 0.0)

        if any(x is None for x in [env, method, model]):
            continue
        if env not in ENVS or method not in METHODS:
            continue

        summary = res.get("summary", {})

        if "mean_ormse" in summary:
            mean = summary["mean_ormse"]
            se = summary.get("se_ormse", 0.0)
        else:
            per_dim = summary.get("per_dim", {})
            if not per_dim:
                continue
            rmses = [v["mean_effect_rmse"] for v in per_dim.values()]
            ses = [v.get("se_effect_rmse", 0.0) for v in per_dim.values()]
            mean = float(np.mean(rmses))
            se = float(np.sqrt(np.mean(np.array(ses) ** 2)))

        rho_key = round(float(rho), 1)
        data[rho_key][env][method][model] = (mean, 1.96 * se)

    return data


def fmt(mean, ci, digits, bold=False):
    s = f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"
    return f"\\textbf{{{s}}}" if bold else s


def build_table(data, rho, model, digits):
    n_methods = len(METHODS)
    col_spec = "l" + "c" * n_methods
    rho_key = round(float(rho), 1)
    label_tag = f"nonadd_{model}_rho{str(rho).replace('.', '')}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Mean ORMSE ($\pm$ 95\% CI) for \textbf{"
        + MODEL_LABEL[model]
        + r"} under \textit{"
        + RHO_LABEL[rho]
        + r"}. Bold = best per row.}",
        f"\\label{{tab:{label_tag}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Function & " + " & ".join(METHOD_LABEL[m] for m in METHODS) + r" \\",
        r"\midrule",
    ]

    for env in ENVS:
        row_vals = {m: data[rho_key][env].get(m, {}).get(model) for m in METHODS}
        valid = {m: v for m, v in row_vals.items() if v is not None}
        best = min(valid, key=lambda m: valid[m][0]) if valid else None

        cells = []
        for m in METHODS:
            entry = row_vals[m]
            if entry is None:
                cells.append("--")
            else:
                mean, ci = entry
                cells.append(fmt(mean, ci, digits, bold=(m == best)))

        lines.append(f"{env} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    args = parse_args()
    data = load_results(args.results_dir)

    if not data:
        print("No results found. Check --results-dir.")
        return

    print(f"Loaded results from: {args.results_dir}")
    for rho in RHOS:
        rho_key = round(float(rho), 1)
        for model in MODELS:
            found = sum(
                1 for env in ENVS for m in METHODS
                if data[rho_key][env].get(m, {}).get(model) is not None
            )
            total = len(ENVS) * len(METHODS)
            print(f"  rho={rho} model={MODEL_LABEL[model]:<4}: {found}/{total} cells filled")

    tables = []
    for rho in RHOS:
        for model in MODELS:
            tables.append(build_table(data, rho, model, args.digits))

    with open(args.outfile, "w") as f:
        f.write("\n\n".join(tables) + "\n")

    n_tables = len(RHOS) * len(MODELS)
    print(f"\nSaved {n_tables} tables -> {args.outfile}")


if __name__ == "__main__":
    main()
