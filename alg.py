import numpy as np
 
 
# helper function: define a grid on the range of the training data
def _make_grid(X_train, d, n_grid):
    lo = X_train[:, d].min()
    hi = X_train[:, d].max()
    return np.linspace(lo, hi, n_grid)
 
 # center the estimated main effect for fair evaluation
def _centre(effects):
    return effects - effects.mean()
 
 # using quantile to define the bin
def _bin_boundaries(data_d, K):
    """Equal-frequency (quantile) bin boundaries — used by ALE and A2D2E."""
    boundaries = np.percentile(data_d, np.linspace(0, 100, K + 1))
    return np.unique(boundaries)           # remove duplicates from ties
 
# define the membership of the bin for each training data
def _bin_members(data_d, boundaries, k):
    """Indices of training points in 1-based bin k."""
    lo, hi = boundaries[k - 1], boundaries[k]
    if k == len(boundaries) - 1:
        return np.where((data_d >= lo) & (data_d <= hi))[0]
    return np.where((data_d >= lo) & (data_d < hi))[0]
 
 
def _bin_index(x_d, boundaries):
    """1-based bin index for scalar x_d (clipped to [1, K])."""
    K = len(boundaries) - 1
    J = int(np.searchsorted(boundaries, x_d, side='right')) - 1
    return max(1, min(J, K))
 
 
# Baseline1: Partial Dependence (PD) Plot
 
def pd_curve(f_hat, X_train, d, n_grid=100):
    N, D   = X_train.shape
    grid   = _make_grid(X_train, d, n_grid)
    effects = np.zeros(n_grid)
 
    for i, val in enumerate(grid):
        # Replace d-th column with val, keep all other columns as-is
        X_pd       = X_train.copy()
        X_pd[:, d] = val
        effects[i] = f_hat(X_pd).mean()
 
    return grid, _centre(effects)
 
 
# Baseline2. Marginal (M) Plot
 
def m_curve(f_hat, X_train, d, n_grid=100, K=40):

    data_d     = X_train[:, d]
    boundaries = _bin_boundaries(data_d, K)
    grid       = _make_grid(X_train, d, n_grid)
    effects    = np.zeros(n_grid)
 
    # Pre-compute f_hat on the full training set (evaluate once)
    f_all = f_hat(X_train)                        # (N,)
 
    for i, val in enumerate(grid):
        k   = _bin_index(val, boundaries)
        I_k = _bin_members(data_d, boundaries, k)
        if len(I_k) == 0:
            # Fall back to global mean if bin is empty
            effects[i] = f_all.mean()
        else:
            effects[i] = f_all[I_k].mean()
 
    return grid, _centre(effects)
 
 
# Baseline3. Accumulated Local Effects (ALE) Plot
 
def ale_curve(f_hat, X_train, d, n_grid=100, K=40):
    N, D       = X_train.shape
    data_d     = X_train[:, d]
    boundaries = _bin_boundaries(data_d, K)
    K_actual   = len(boundaries) - 1
 
    # ── Compute per-bin increments delta_k ────────────────────────────────────
    delta = np.zeros(K_actual)           # delta[k-1] = increment for bin k
 
    for k in range(1, K_actual + 1):
        I_k = _bin_members(data_d, boundaries, k)
        if len(I_k) == 0:
            delta[k - 1] = 0.0
            continue
 
        X_lo       = X_train[I_k].copy()
        X_hi       = X_train[I_k].copy()
        X_lo[:, d] = boundaries[k - 1]   # lower boundary z_d^k
        X_hi[:, d] = boundaries[k]       # upper boundary z_d^{k+1}
 
        delta[k - 1] = (f_hat(X_hi) - f_hat(X_lo)).mean()
 
    # ── Evaluate on grid by accumulating deltas up to J(x_d) ─────────────────
    grid    = _make_grid(X_train, d, n_grid)
    effects = np.zeros(n_grid)
 
    for i, val in enumerate(grid):
        J          = _bin_index(val, boundaries)
        effects[i] = delta[:J].sum()
 
    return grid, _centre(effects)
 
 
# Proposed A2D2E
 
def _d_optimal_design(delta, x_n, D):
    """
    All 2^D vertices of the hypercube centered at x_n with edge delta.
    Returns V (absolute), V_tilde (shifted = V - x_n).
    """
    signs   = np.array([[((i >> b) & 1) * 2 - 1 for b in range(D)]
                        for i in range(2 ** D)], dtype=float)
    V_tilde = (delta / 2.0) * signs
    V       = x_n[None, :] + V_tilde
    return V, V_tilde
 
 
def _estimate_beta(f_hat, V, V_tilde, d, delta, D):
    """
    Fast closed-form OLS slope for variable d:
        beta_d = (V_tilde^T y)[d] / (2^{D-2} * delta^2)
    """
    y    = f_hat(V)
    return float(np.dot(V_tilde[:, d], y)) / (2 ** (D - 2) * delta ** 2)
 
 
def a2d2e_curve(f_hat, X_train, d, n_grid=100, K=40, delta=0.01):

    N, D       = X_train.shape
    data_d     = X_train[:, d]
    boundaries = _bin_boundaries(data_d, K)
    K_actual   = len(boundaries) - 1
 
    # ── Per-bin slopes ────────────────────────────────────────────────────────
    beta_k = np.zeros(K_actual)
 
    for k in range(1, K_actual + 1):
        I_k = _bin_members(data_d, boundaries, k)
        if len(I_k) == 0:
            continue
 
        beta_sum = 0.0
        for n in I_k:
            V, V_tilde   = _d_optimal_design(delta, X_train[n], D)
            beta_sum    += _estimate_beta(f_hat, V, V_tilde, d, delta, D)
        beta_k[k - 1] = beta_sum / len(I_k)
 
    # ── Evaluate on grid ──────────────────────────────────────────────────────
    grid    = _make_grid(X_train, d, n_grid)
    effects = np.zeros(n_grid)
 
    for i, val in enumerate(grid):
        J = _bin_index(val, boundaries)
        for k in range(1, J + 1):
            bin_width    = boundaries[k] - boundaries[k - 1]
            effects[i]  += bin_width * beta_k[k - 1]
 
    return grid, _centre(effects)
 
 
# 5. DALE (Differential ALE)
 
def dale_curve(f_hat, X_train, d, n_grid=100, K=40,
               grad_fn=None, eps=1e-4):

    N, D       = X_train.shape
    data_d     = X_train[:, d]
    boundaries = _bin_boundaries(data_d, K)
    K_actual   = len(boundaries) - 1
 
    # ── Partial derivative df/dx_d at every training point ───────────────────
    if grad_fn is not None:
        # Analytic gradient (e.g. torch autograd, GP posterior gradient)
        grads = grad_fn(X_train)                        # (N,)
    else:
        # Central finite differences: (f(x + eps*e_d) - f(x - eps*e_d)) / 2eps
        X_hi       = X_train.copy(); X_hi[:, d] += eps
        X_lo       = X_train.copy(); X_lo[:, d] -= eps
        grads      = (f_hat(X_hi) - f_hat(X_lo)) / (2.0 * eps)  # (N,)
 
    # ── Per-bin average gradient (= estimated local slope) ───────────────────
    avg_grad = np.zeros(K_actual)
 
    for k in range(1, K_actual + 1):
        I_k = _bin_members(data_d, boundaries, k)
        if len(I_k) == 0:
            avg_grad[k - 1] = 0.0
        else:
            avg_grad[k - 1] = grads[I_k].mean()
 
    # ── Accumulate (bin_width * avg_grad_k) up to J(x_d) ─────────────────────
    grid    = _make_grid(X_train, d, n_grid)
    effects = np.zeros(n_grid)
 
    for i, val in enumerate(grid):
        J = _bin_index(val, boundaries)
        for k in range(1, J + 1):
            effects[i] += (boundaries[k] - boundaries[k - 1]) * avg_grad[k - 1]
 
    return grid, _centre(effects)
 
