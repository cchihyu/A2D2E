# wall_clock — Appendix C: Wall-Clock Complexity

Reproduces the **wall-clock timing tables** in Appendix C: empirical runtime as a function of training-set size N and dimensionality D for each estimator.

## Script

`run.py` is fully self-contained — it does not depend on `src/` or `model_params/`. It uses a 1-nearest-neighbour surrogate and measures wall-clock time for PD, M, ALE, DALE, and A2D2E across a grid of (N, D) settings.

## Usage

```bash
python experiments/wall_clock/run.py --outfile complexity.tex
```

This writes a LaTeX table directly to `complexity.tex`.

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--outfile` | Path for the output LaTeX file | `complexity.tex` |

No GPU or pre-tuned parameters are required.
