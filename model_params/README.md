# model_params — Pre-tuned Hyperparameters

This directory contains hyperparameter files for each combination of benchmark function and model type. These were tuned offline and are loaded at runtime by the experiment scripts.

## File naming

```
{env}_{model}.py
```

- **env**: `f0`, `f1`, `f2`, `f3`, `f4`, `f5`, `f6`
- **model**: `knn`, `nn`, `rf`, `gp`

28 files total (7 environments × 4 models).

## Format

Each file defines a single `PARAMS` dictionary, for example:

```python
PARAMS = {
    "n_neighbors": 10,
}
```

The dictionary keys match the constructor arguments of the corresponding sklearn estimator. They are loaded via `src/models.py:load_params(path)`.

## Usage

Pass the directory path with `--param-dir model_params` when running experiment scripts. The runner constructs the file path as `{param_dir}/{env}_{model}.py`.
