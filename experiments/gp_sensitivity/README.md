# gp_sensitivity — Figure 3: GPR Length-Scale Sensitivity

Reproduces **Figure 3** from the paper: how the GPR length-scale hyperparameter affects A2D2E's ORMSE estimate on a one-dimensional synthetic example.

## Script

`run.py` is self-contained — it does not use `src/` or `model_params/`.

## Usage

```bash
# Run one (length-scale index, replication) pair
python experiments/gp_sensitivity/run.py \
    --ls_idx 0 --rep 0 --out_dir results_gp_sensitivity

# Run all 15 length scales × 100 replications
for ls_idx in $(seq 0 14); do
  for rep in $(seq 0 99); do
    python experiments/gp_sensitivity/run.py \
        --ls_idx $ls_idx --rep $rep --out_dir results_gp_sensitivity
  done
done
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--ls_idx` | Index into the 15-point length-scale grid (0–14) |
| `--rep` | Replication index (0–99) |
| `--out_dir` | Directory for output JSON files |

## Output

One JSON file per `(ls_idx, rep)` pair in `--out_dir`. Post-processing and plotting are done separately (not included here; results are assembled into Figure 3 manually).
