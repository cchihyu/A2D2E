# nonadditive — Appendix D.2: Non-Additive Benchmarks (R)

Reproduces **Appendix D.2** from the paper: main-effect estimation on ten non-additive benchmark functions using KNN, neural network, random forest, GP, and GBM surrogates fitted in R.

## Files

| File | Purpose |
|------|---------|
| `A2D2E_main.Rmd` | Main experiment notebook — knit in RStudio or run chunk by chunk |
| `A2D2E_main_function.R` | R implementation of the A2D2E estimator |
| `compute_truth.R` | Compute ground-truth main effects via numerical integration |
| `real_data.R` | Appendix D.3: real data analysis |
| `ground_truth/` | Generated ground-truth files (created by `compute_truth.R`, not tracked in git) |

## Requirements

Install the following R packages before running:

```r
install.packages(c("ALEPlot", "MASS", "nnet", "DiceKriging",
                   "e1071", "randomForest", "cubature", "gbm3"))
```

`gbm3` may need to be installed from GitHub: `remotes::install_github("gbm-developers/gbm3")`.

## Running

**Step 1 — Generate ground truth** (run once from `experiments/nonadditive/`):

```bash
cd experiments/nonadditive
Rscript compute_truth.R
```

This writes `ground_truth/*.txt` files via numerical integration.

**Step 2 — Run the main experiment:**

Open `A2D2E_main.Rmd` in RStudio and knit, or source it chunk by chunk. The working directory must be `experiments/nonadditive/` so that relative paths resolve correctly.

## Benchmark functions

| Name | D | Description |
|------|---|-------------|
| simple | 2 | x₁² + x₂ |
| simple2 | 4 | x₁x₂ − x₂x₃ + x₄x₁ |
| franke2d | 2 | Franke function |
| braninsc | 2 | Scaled Branin |
| grlee09 | 6 | exp(sin(·)) + x₂x₃ + x₄ |
| levy | 6 | Levy function |
| ackley | 6 | Ackley function |
| fried | 5 | Friedman function |
| detpep108d | 8 | Det-pep function |
| f_norm | 8 | Borehole function |
