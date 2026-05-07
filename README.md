# A2D2E: Accumulated Aggregated D-Optimal Designs for Estimating Main Effects

Code for the paper *"Accumulated Aggregated D-Optimal Designs for Estimating Main Effects in Black-Box Models"* (NeurIPS 2026).

## Repository structure

```
a2d2e/
├── src/                         # Shared library
│   ├── alg.py                   # PD, M, ALE, DALE, A2D2E algorithms
│   ├── env.py                   # Additive benchmark functions f0–f6
│   ├── env_nonadditive.py       # Non-additive functions g1–g7 with analytical ALE
│   └── md.py                    # Model fitting (KNN, NN, RF, GP) + gradient utilities
│
├── param/                       # Pre-tuned hyperparameters for f0–f6 × {knn,nn,rf,gp}
├── param_nonadditive/           # Pre-tuned hyperparameters for g1–g7 × {knn,nn,rf,gp}
│
├── experiments/
│   ├── table2/
│   │   ├── run.py               # Table 2 experiment runner (one job per config)
│   │   └── summary.py           # Aggregate results → LaTeX table
│   ├── figure3/
│   │   └── run.py               # Figure 3: GPR length-scale sensitivity (self-contained)
│   ├── appendix_c/
│   │   └── run.py               # Appendix C: wall-clock complexity tables (self-contained)
│   └── appendix_d2/
│       ├── run.py               # Appendix D.2: non-additive benchmark runner
│       └── summary.py           # Non-additive LaTeX tables
│
├── hpc/
│   ├── run.pbs                  # PBS job array for Table 2 (420 jobs)
│   ├── run_nonadditive.pbs      # PBS job array for Appendix D.2 (280 jobs)
│   └── submit_all.sh            # Submit all HPC jobs
│
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Running experiments

### Table 2 (main results)

**Single run** (one env/method/model/dependence/noise combination):

```bash
python experiments/table2/run.py \
    --env f0 --method a2d2e --model knn \
    --dependence independent --n-train 300 --n-reps 100 \
    --noise-frac 0.1 --K 40 --delta 0.025 \
    --param-dir param --outdir results_new --verbose
```

**Generate LaTeX table** after all runs complete:

```bash
python experiments/table2/summary.py \
    --results-dir results_new --outfile table2.tex
```

**On HPC** (Imperial CX3, PBS):

```bash
bash hpc/submit_all.sh
```

### Figure 3 (GPR length-scale sensitivity)

```bash
# Run one (ls_idx, rep) pair — e.g. first length scale, rep 0
python experiments/figure3/run.py --ls_idx 0 --rep 0 --out_dir results_spike

# Run all 15 length scales × 100 reps in a loop
for ls_idx in $(seq 0 14); do
  for rep in $(seq 0 99); do
    python experiments/figure3/run.py --ls_idx $ls_idx --rep $rep --out_dir results_spike
  done
done
```

### Appendix C (wall-clock complexity)

Self-contained; no pre-tuned params needed:

```bash
python experiments/appendix_c/run.py --outfile complexity.tex
```

### Appendix D.2 (non-additive benchmarks)

```bash
python experiments/appendix_d2/run.py \
    --env g1 --method a2d2e_goldilocks --model knn \
    --rho 0.7 --n-train 200 --n-reps 30 --noise-frac 0.3 \
    --param-dir param_nonadditive --outdir results_nonadditive --verbose

python experiments/appendix_d2/summary.py \
    --results-dir results_nonadditive --outfile table_nonadditive.tex
```

## Benchmark functions

### Additive (Table 2)

| ID | D  | Description |
|----|----|-------------|
| f0 | 3  | x₁ + x₂² + 0·x₃ |
| f1 | 2  | sin(u₁) + u₂² |
| f2 | 4  | sin(10u₁) + sin(u₂) + u₃³−u₃ + sigmoid(10u₄) |
| f3 | 4  | kink, sqrt-sign, sin(πu/2), u·log(\|u\|+1) |
| f4 | 4  | 10·sin(u₁) + 0.1·u₂² + 5·exp(−u₃²) + 0.05·u₄ |
| f5 | 8  | Multi-scale oscillation |
| f6 | 10 | Fourier + polynomial + rational |

### Non-additive (Appendix D.2, X ~ N(0, Σ_AR1(ρ)))

| ID | D  | Description |
|----|----|-------------|
| g1 | 2  | Bilinear: x₁·x₂ |
| g2 | 4  | Squared sum: (x₁+⋯+x₄)² |
| g3 | 4  | Exp sum: exp((x₁+⋯+x₄)/4) |
| g4 | 3  | Ishigami |
| g5 | 4  | Quadratic form: xᵀAx |
| g6 | 4  | Cyclic polynomial |
| g7 | 8  | Detpep |

## Competing methods

| Method | Description |
|--------|-------------|
| PD     | Partial Dependence |
| M      | Marginal (M-plot) |
| ALE    | Accumulated Local Effects |
| DALE   | Differential ALE |
| A2D2E  | **Ours**: Accumulated Aggregated D-Optimal Designs |

## Evaluation metric

ORMSE = (1/D) Σ_d RMSE(estimated main effect d, true main effect d), averaged over 100 replications.
