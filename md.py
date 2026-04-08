import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


# ── Cross-validation setup ────────────────────────────────────────────────────
N_SPLITS   = 10      # inner CV folds
RANDOM_STATE = 42


def _cv(n):
    return KFold(n_splits=min(N_SPLITS, n // 5 + 1),
                 shuffle=True, random_state=RANDOM_STATE)


# KNN

def fit_knn(X, y, verbose=False):

    N = len(y)
    k_max = max(3, min(50, N // 5))
    k_grid = sorted(set(
        [1, 3, 5, 7, 10, 15, 20, 30, 50] + list(range(3, k_max + 1, 5))
    ))
    k_grid = [k for k in k_grid if k <= N - 1]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsRegressor()),
    ])
    search = GridSearchCV(
        pipe,
        {"knn__n_neighbors": k_grid,
         "knn__weights":     ["uniform", "distance"]},
        cv=_cv(N), scoring="neg_mean_squared_error",
        n_jobs=-1, refit=True,
    )
    search.fit(X, y)

    if verbose:
        bp = search.best_params_
        print(f"[KNN] best k={bp['knn__n_neighbors']}, "
              f"weights={bp['knn__weights']}, "
              f"CV RMSE={(-search.best_score_)**0.5:.4f}")

    model = search.best_estimator_
    return lambda Xq: model.predict(np.atleast_2d(Xq))


# Neural Network (MLP)

def fit_nn(X, y, verbose=False):

    N, D = X.shape
    units = [max(8, D * 2), max(16, D * 4), max(32, D * 8), 64, 128]
    param_grid = {
        "mlp__hidden_layer_sizes": (
            [(u,)       for u in units] +
            [(u, u)     for u in units] +
            [(u, u // 2) for u in units if u // 2 >= 4]
        ),
        "mlp__alpha":              [1e-4, 1e-3, 1e-2, 1e-1],
        "mlp__learning_rate_init": [1e-3, 5e-3, 1e-2],
    }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPRegressor(
            activation="relu",
            solver="adam",
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )),
    ])
    search = GridSearchCV(
        pipe, param_grid,
        cv=_cv(N), scoring="neg_mean_squared_error",
        n_jobs=-1, refit=True,
    )
    search.fit(X, y)

    model = search.best_estimator_

    if verbose:
        bp = search.best_params_
        print(f"[NN]  best layers={bp['mlp__hidden_layer_sizes']}, "
              f"alpha={bp['mlp__alpha']}, lr={bp['mlp__learning_rate_init']}, "
              f"CV RMSE={(-search.best_score_)**0.5:.4f}")

    f_hat = lambda Xq: model.predict(np.atleast_2d(Xq))
    return model, f_hat


# Random Forest

def fit_rf(X, y, verbose=False):
  
    N, D = X.shape
    param_grid = {
        "max_depth":        [None, 3, 5, 8, 12, 20],
        "min_samples_leaf": [1, 2, 4, 8, 16],
        "max_features":     ["sqrt", "log2", 0.5, 1.0],
    }
    base = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search = GridSearchCV(
        base, param_grid,
        cv=_cv(N), scoring="neg_mean_squared_error",
        n_jobs=1,   # outer loop sequential since RF already uses n_jobs=-1
        refit=True,
    )
    search.fit(X, y)

    if verbose:
        bp = search.best_params_
        print(f"[RF]  best depth={bp['max_depth']}, "
              f"min_leaf={bp['min_samples_leaf']}, "
              f"max_feat={bp['max_features']}, "
              f"CV RMSE={(-search.best_score_)**0.5:.4f}")

    model = search.best_estimator_
    return lambda Xq: model.predict(np.atleast_2d(Xq))


# Gaussian Process

def fit_gp(X, y, verbose=False):
    N, D = X.shape

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    y_mean = y.mean()
    y_std  = y.std() if y.std() > 0 else 1.0
    y_sc   = (y - y_mean) / y_std

    # ARD kernel: one length_scale per input dimension
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=np.ones(D),
              length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1,
                      noise_level_bounds=(1e-5, 1e1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=False,          # we normalise manually above
        random_state=RANDOM_STATE,
    )
    gp.fit(X_sc, y_sc)

    if verbose:
        print(f"[GP]  optimised kernel: {gp.kernel_}")
        print(f"      log-marginal-likelihood: {gp.log_marginal_likelihood_value_:.4f}")

    def f_hat(Xq):
        Xq_sc = scaler.transform(np.atleast_2d(Xq))
        return gp.predict(Xq_sc) * y_std + y_mean

    return gp, f_hat, scaler, y_std



def get_nn_gradient(nn_pipeline, X, d):

    X      = np.atleast_2d(X)
    scaler = nn_pipeline.named_steps["scaler"]
    mlp    = nn_pipeline.named_steps["mlp"]

    X_sc   = scaler.transform(X)              # (N, D)
    coefs  = mlp.coefs_                       # list: W_l shape (in, out)
    n_layers = len(coefs)                     # number of weight matrices

    # ── Forward pass: store pre-activations z_l for hidden layers ─────────────
    h  = X_sc.copy()                          # (N, D)
    zs = []
    for l in range(n_layers - 1):             # hidden layers
        z = h @ coefs[l] + mlp.intercepts_[l]
        zs.append(z)                          # (N, units_l)
        h = np.maximum(z, 0.0)               # ReLU

    # ── Backward pass ────────────────────────────────────────────────────────
    N = X_sc.shape[0]
    J = np.tile(coefs[-1][:, 0], (N, 1))      # (N, h_{L-1}) — always 2D
    for l in reversed(range(n_layers - 1)):
        relu_mask = (zs[l] > 0).astype(float) # (N, units_l)
        J = relu_mask * J                      # (N, units_l)
        J = J @ coefs[l].T                     # (N, in_dim_of_layer_l)
    # J is now (N, D): d_out / d_x_sc

    # ── Chain rule through the StandardScaler ─────────────────────────────────
    # x_sc_d = (x_d - mu_d) / scale_d  =>  d x_sc_d / d x_d = 1/scale_d
    grads = J[:, d] / scaler.scale_[d]
    return grads


def get_gp_gradient(gp, scaler, y_std, X, d):
    X    = np.atleast_2d(X)
    X_sc = scaler.transform(X)            # scale to GP's input space

    # ── Extract kernel parameters ─────────────────────────────────────────────
    # gp.kernel_ is ConstantKernel * RBF + WhiteKernel after fitting
    # We need the RBF component's length scales
    k         = gp.kernel_
    const_k   = k.k1.k1           # ConstantKernel
    rbf_k     = k.k1.k2           # RBF with ARD length scales
    C         = const_k.constant_value
    ell       = rbf_k.length_scale # array of shape (D,)

    X_train   = gp.X_train_        # training inputs in scaled space (N_tr, D)
    alpha     = gp.alpha_          # (K + sigma^2 I)^{-1} y, shape (N_tr,)

    # ── Compute kernel matrix k(X*, X_train) ─────────────────────────────────
    # Shape: (N_query, N_train)
    diff      = X_sc[:, None, :] - X_train[None, :, :]  # (N_q, N_tr, D)
    sq_dist   = np.sum((diff / ell) ** 2, axis=-1)       # (N_q, N_tr)
    K_star    = C * np.exp(-0.5 * sq_dist)               # (N_q, N_tr)

    # ── Gradient of k(x*, x_i) w.r.t. x*_d (scaled space) ───────────────────
    # d k / d x*_d = -(x*_d - x_{i,d}) / l_d^2 * k(x*, x_i)
    dK_dx_d   = -diff[:, :, d] / (ell[d] ** 2) * K_star  # (N_q, N_tr)

    # ── Gradient of posterior mean in scaled space ────────────────────────────
    dmu_dx_sc = dK_dx_d @ alpha                            # (N_q,)

    # ── Chain rule: transform gradient to original space ─────────────────────
    # x_sc = (x - mu_sc) / std_sc  =>  dx_sc/dx_d = 1/std_sc[d]
    dmu_dx_d  = dmu_dx_sc / scaler.scale_[d]

    # ── Re-scale from normalised y back to original y ─────────────────────────
    return dmu_dx_d * y_std

def get_gradient(f_hat, X, d, eps=1e-4):
    X = np.atleast_2d(X)
    X_hi = X.copy(); X_hi[:, d] += eps
    X_lo = X.copy(); X_lo[:, d] -= eps
    return (f_hat(X_hi) - f_hat(X_lo)) / (2.0 * eps)


def fit_all(X, y, verbose=False):
    gp, f_hat_gp, scaler, y_std = fit_gp(X, y, verbose=verbose)
    nn_pipe, f_hat_nn = fit_nn(X, y, verbose=verbose)
    f_hat_knn = fit_knn(X, y, verbose=verbose)
    f_hat_rf  = fit_rf(X, y, verbose=verbose)

    return {
        "knn": {
            "f_hat": f_hat_knn,
            "grad_fn_builder": lambda d: (lambda X_: get_gradient(f_hat_knn, X_, d, eps=1e-4)),
        },
        "nn": {
            "f_hat": f_hat_nn,
            "grad_fn_builder": lambda d: (lambda X_: get_nn_gradient(nn_pipe, X_, d)),
        },
        "rf": {
            "f_hat": f_hat_rf,
            "grad_fn_builder": lambda d: (lambda X_: get_gradient(f_hat_rf, X_, d, eps=1e-2)),
        },
        "gp": {
            "f_hat": f_hat_gp,
            "grad_fn_builder": lambda d: (lambda X_: get_gp_gradient(gp, scaler, y_std, X_, d)),
        },
    }
