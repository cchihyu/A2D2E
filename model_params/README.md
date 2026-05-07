# model_params — Pre-tuned Hyperparameters

This directory contains KNN hyperparameter files for each of the seven benchmark environments. These were tuned offline and are loaded at runtime by the experiment scripts.

## File naming

```
{env}_knn.py
```

- **env**: `f0`, `f1`, `f2`, `f3`, `f4`, `f5`, `f6`

7 files total (one per environment).

## Format

Each file defines a single `PARAMS` dictionary, for example:

```python
PARAMS = {
    "n_neighbors": 10,
}
```

The dictionary keys match the constructor arguments of the `sklearn` KNN regressor. They are loaded via `src/models.py:load_params(path)`.

## Usage

Pass the directory path with `--param-dir model_params` when running experiment scripts. The runner constructs the file path as `{param_dir}/{env}_knn.py`.
