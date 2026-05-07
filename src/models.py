import importlib.util
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def load_params(param_file):
    """Load a Python parameter file that defines a top-level dict named PARAMS."""
    param_file = Path(param_file)
    spec = importlib.util.spec_from_file_location(param_file.stem, param_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PARAMS"):
        raise ValueError(f"Parameter file '{param_file}' does not define PARAMS")

    params = module.PARAMS
    if not isinstance(params, dict):
        raise ValueError(f"PARAMS in '{param_file}' must be a dict")
    return params


def _extract_best_params(params, expected_model=None):
    if params is None:
        return {}

    if expected_model is not None and isinstance(params, dict):
        model_name = params.get("model")
        if model_name is not None and model_name != expected_model:
            raise ValueError(
                f"Expected params for model '{expected_model}', got '{model_name}'"
            )

    if isinstance(params, dict) and "best_params" in params:
        return dict(params["best_params"])
    if isinstance(params, dict):
        return dict(params)

    raise TypeError("params must be a dict or a payload containing 'best_params'")


def fit_knn(X, y, params=None, verbose=False):
    best_params = _extract_best_params(params, expected_model="knn")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(
            n_neighbors=best_params.get("knn__n_neighbors", 5),
            weights=best_params.get("knn__weights", "uniform"),
        )),
    ])
    model.fit(X, y)

    if verbose:
        print(f"[KNN] k={model.named_steps['knn'].n_neighbors}, "
              f"weights={model.named_steps['knn'].weights}")

    return lambda Xq: model.predict(np.atleast_2d(Xq))


def fit_nn(X, y, params=None, verbose=False):
    best_params = _extract_best_params(params, expected_model="nn")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            activation="relu",
            solver="adam",
            hidden_layer_sizes=best_params.get("mlp__hidden_layer_sizes", (64,)),
            alpha=best_params.get("mlp__alpha", 1e-4),
            learning_rate_init=best_params.get("mlp__learning_rate_init", 1e-3),
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(X, y)

    if verbose:
        mlp = pipe.named_steps["mlp"]
        print(f"[NN] layers={mlp.hidden_layer_sizes}, "
              f"alpha={mlp.alpha}, lr={mlp.learning_rate_init}")

    f_hat = lambda Xq: pipe.predict(np.atleast_2d(Xq))
    return pipe, f_hat


def fit_rf(X, y, params=None, verbose=False):
    best_params = _extract_best_params(params, expected_model="rf")

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=best_params.get("max_depth", None),
        min_samples_leaf=best_params.get("min_samples_leaf", 1),
        max_features=best_params.get("max_features", 1.0),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)

    if verbose:
        print(f"[RF] depth={model.max_depth}, "
              f"min_leaf={model.min_samples_leaf}, "
              f"max_feat={model.max_features}")

    return lambda Xq: model.predict(np.atleast_2d(Xq))


def fit_gp(X, y, params=None, verbose=False):
    best_params = _extract_best_params(params, expected_model="gp")

    _, D = X.shape
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    y_mean = y.mean()
    y_std = y.std() if y.std() > 0 else 1.0
    y_sc = (y - y_mean) / y_std

    constant_value = best_params.get("constant_value", 1.0)
    length_scale = np.asarray(best_params.get("length_scale", np.ones(D)), dtype=float)
    noise_level = best_params.get("noise_level", 0.1)

    if length_scale.shape == ():
        length_scale = np.repeat(float(length_scale), D)
    if len(length_scale) != D:
        raise ValueError(f"GP length_scale has length {len(length_scale)}, but X has D={D}")

    kernel = (
        ConstantKernel(constant_value=constant_value, constant_value_bounds="fixed")
        * RBF(length_scale=length_scale, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=noise_level, noise_level_bounds="fixed")
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=False,
        random_state=RANDOM_STATE,
    )
    gp.fit(X_sc, y_sc)

    if verbose:
        print(f"[GP] kernel={gp.kernel_}")

    def f_hat(Xq):
        Xq_sc = scaler.transform(np.atleast_2d(Xq))
        return gp.predict(Xq_sc) * y_std + y_mean

    return gp, f_hat, scaler, y_std


def get_gradient(f_hat, X, d, eps=1e-4):
    """Central finite-difference gradient of f_hat w.r.t. x_d."""
    X = np.atleast_2d(X)
    X_hi = X.copy(); X_hi[:, d] += eps
    X_lo = X.copy(); X_lo[:, d] -= eps
    return (f_hat(X_hi) - f_hat(X_lo)) / (2.0 * eps)


def get_nn_gradient(nn_pipeline, X, d):
    """Analytic gradient of the sklearn MLP prediction w.r.t. x_d."""
    X = np.atleast_2d(X)
    scaler = nn_pipeline.named_steps["scaler"]
    mlp = nn_pipeline.named_steps["mlp"]

    X_sc = scaler.transform(X)
    coefs = mlp.coefs_
    n_layers = len(coefs)

    h = X_sc.copy()
    zs = []
    for l in range(n_layers - 1):
        z = h @ coefs[l] + mlp.intercepts_[l]
        zs.append(z)
        h = np.maximum(z, 0.0)

    N = X_sc.shape[0]
    J = np.tile(coefs[-1][:, 0], (N, 1))
    for l in reversed(range(n_layers - 1)):
        relu_mask = (zs[l] > 0).astype(float)
        J = relu_mask * J
        J = J @ coefs[l].T

    return J[:, d] / scaler.scale_[d]


def get_gp_gradient(gp, scaler, y_std, X, d):
    """Analytic gradient of the GP posterior mean w.r.t. x_d."""
    X = np.atleast_2d(X)
    X_sc = scaler.transform(X)

    k = gp.kernel_
    const_k = k.k1.k1
    rbf_k = k.k1.k2
    C = const_k.constant_value
    ell = np.asarray(rbf_k.length_scale, dtype=float)

    X_train = gp.X_train_
    alpha = gp.alpha_

    diff = X_sc[:, None, :] - X_train[None, :, :]
    sq_dist = np.sum((diff / ell) ** 2, axis=-1)
    K_star = C * np.exp(-0.5 * sq_dist)

    dK_dx_d = -diff[:, :, d] / (ell[d] ** 2) * K_star
    dmu_dx_sc = dK_dx_d @ alpha
    dmu_dx_d = dmu_dx_sc / scaler.scale_[d]

    return dmu_dx_d * y_std
