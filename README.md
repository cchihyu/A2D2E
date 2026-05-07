# A2D2E: Accumulated Aggregated D-Optimal Designs for Estimating Main Effects

Code for the paper *"Accumulated Aggregated D-Optimal Designs for Estimating Main Effects in Black-Box Models"* (NeurIPS 2026).

## Repository structure

```
a2d2e/
├── src/                              # Shared Python library
│   ├── algorithms.py                 # PD, M, ALE, DALE, A2D2E curve estimators
│   ├── benchmarks.py                 # Additive benchmark functions f0–f6 + sampling
│   └── models.py                     # Model fitting (KNN, NN, RF, GP) + gradients
│
├── model_params/                     # Pre-tuned hyperparameters for f0–f6 × {knn,nn,rf,gp}
│
├── experiments/
│   ├── main_comparison/              # Table 2: ORMSE comparison across methods
│   │   ├── run.py                    # Experiment runner (one job per configuration)
│   │   └── summarize.py             # Aggregate results → LaTeX table
│   ├── gp_sensitivity/               # Figure 3: GPR length-scale sensitivity (self-contained)
│   │   └── run.py
│   ├── wall_clock/                   # Appendix C: wall-clock complexity (self-contained)
│   │   └── run.py
│   └── nonadditive/                  # Appendix D.2: non-additive benchmarks (R)
│       ├── A2D2E_main.Rmd            # Main experiment notebook
│       ├── A2D2E_main_function.R     # A2D2E R implementation
│       ├── compute_truth.R           # Ground truth via numerical integration
│       ├── real_data.R               # Appendix D.3: real data analysis
│       └── ground_truth/             # Pre-computed ground truth .txt files
│
├── requirements.txt
└── pyrightconfig.json
```

## Installation

```bash
pip install -r requirements.txt
```

## Running experiments

### Table 2 (main comparison)

**Single run** (one env/method/model/dependence/noise combination):

```bash
python experiments/main_comparison/run.py \
    --env f0 --method a2d2e --model knn \
    --dependence independent --n-train 300 --n-reps 100 \
    --noise-frac 0.1 --K 40 --delta 0.025 \
    --param-dir model_params --outdir results --verbose
```

**Generate LaTeX table** after all runs complete:

```bash
python experiments/main_comparison/summarize.py \
    --results-dir results --outfile table2.tex
```

**On HPC** (Imperial CX3, PBS):

```bash
bash hpc/submit_all.sh
```

### Figure 3 (GPR length-scale sensitivity)

```bash
# Run one (ls_idx, rep) pair
python experiments/gp_sensitivity/run.py --ls_idx 0 --rep 0 --out_dir results_gp_sensitivity

# Run all 15 length scales × 100 reps
for ls_idx in $(seq 0 14); do
  for rep in $(seq 0 99); do
    python experiments/gp_sensitivity/run.py --ls_idx $ls_idx --rep $rep --out_dir results_gp_sensitivity
  done
done
```

### Appendix C (wall-clock complexity)

Self-contained; no pre-tuned params needed:

```bash
python experiments/wall_clock/run.py --outfile complexity.tex
```

### Appendix D.2 (non-additive benchmarks, R)

First generate the ground-truth main effects (numerical integration, run once):

```bash
cd experiments/nonadditive
Rscript compute_truth.R
```

Then open `A2D2E_main.Rmd` in RStudio and knit, or run chunk by chunk.

Required R packages: `ALEPlot`, `MASS`, `nnet`, `DiceKriging`, `e1071`, `randomForest`, `cubature`, `gbm3`

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

### Non-additive (Appendix D.2)

| Function   | D | Description |
|------------|---|-------------|
| simple     | 2 | x₁² + x₂ |
| simple2    | 4 | x₁x₂ − x₂x₃ + x₄x₁ |
| franke2d   | 2 | Franke function |
| braninsc   | 2 | Scaled Branin |
| grlee09    | 6 | exp(sin(·)) + x₂x₃ + x₄ |
| levy       | 6 | Levy function |
| ackley     | 6 | Ackley function |
| fried      | 5 | Friedman function |
| detpep108d | 8 | Det-pep function |
| f_norm     | 8 | Borehole function |

## Competing methods

| Method | Description |
|--------|-------------|
| PD     | Partial Dependence |
| M      | Marginal (M-plot) |
| ALE    | Accumulated Local Effects |
| DALE   | Differential ALE |
| A2D2E  | **Ours**: Accumulated Aggregated D-Optimal Designs |

## Evaluation metric

ORMSE = (1/D) Σ_d RMSE(estimated main effect d, true main effect d), averaged over replications.
