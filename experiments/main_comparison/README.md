# main_comparison — Table 2: ORMSE Comparison

Reproduces **Table 2** from the paper: ORMSE (Overall RMSE) for PD, M, ALE, DALE, and A2D2E across seven additive benchmark functions, four dependence levels, and three noise fractions.

## Scripts

| Script | Purpose |
|--------|---------|
| `run.py` | Run one experiment configuration (env × method × model × dependence × noise) |
| `summarize.py` | Aggregate all result JSON files into a LaTeX table |

## Quick start

```bash
# From the repository root
python experiments/main_comparison/run.py \
    --env f0 --method a2d2e --model knn \
    --dependence independent --n-train 300 --n-reps 100 \
    --noise-frac 0.1 --K 40 --delta 0.025 \
    --param-dir model_params --outdir results --verbose
```

## Key arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--env` | Benchmark function (`f0`–`f6`) | required |
| `--method` | Estimator (`pd`, `m`, `ale`, `dale`, `a2d2e`) | required |
| `--model` | Surrogate model (`knn`, `nn`, `rf`, `gp`) | required |
| `--dependence` | Feature dependence (`independent`, `low`, `high`, `veryhigh`) | `independent` |
| `--n-train` | Training set size | required |
| `--n-reps` | Number of Monte Carlo replications | required |
| `--noise-frac` | Noise level as fraction of signal std | `0.1` |
| `--K` | Number of bins | `40` |
| `--delta` | Half-width of A2D2E perturbation | `0.025` |
| `--param-dir` | Directory with hyperparameter files | `model_params` |
| `--outdir` | Output directory for JSON results | `results` |
| `--save-curves` | Also save estimated and true curves to JSON | off |

## Output

Each run writes a JSON file to `--outdir`:

```
results/env-f0_method-a2d2e_model-knn_dep-independent_n-300_reps-100_noise-0.1_seed-0.json
```

The JSON contains per-replication ORMSE values and a summary with mean ± SE.

## Generating Table 2

After all 420 jobs complete:

```bash
python experiments/main_comparison/summarize.py \
    --results-dir results --outfile table2.tex
```

## HPC

The full sweep (7 envs × 5 methods × 1 model × 3 noises × 4 dependences = 420 jobs) runs as a PBS job array. See `hpc/`.
