"""
Non-additive benchmark environments with closed-form ALE under MVN.

X ~ N(0, Sigma) where Sigma is AR(1): Sigma[i,j] = rho^|i-j|

Interface
---------
    ENVS                                  : dict of env dicts (g1-g7)
    sample(env, n, rho, noise_frac, seed) : -> X, y
    true_effect_on_grid(env, d, grid, rho): -> array (n_grid,)

Functions
---------
    g1  D=2  f = x1*x2
    g2  D=4  f = (x1+x2+x3+x4)^2
    g3  D=4  f = exp((x1+x2+x3+x4)/4)
    g4  D=3  f = sin(x1) + 7*sin^2(x2) + 0.1*x3^4*sin(x1)   (Ishigami)
    g5  D=4  f = x^T A x
    g6  D=4  f = x1^2*x2 + x2^2*x3 + x3^2*x4 + x4^2*x1
    g7  D=8  f = sum_{i=1}^8 (sum_{j<=i} x_j - i/2)^2
"""

import numpy as np


# ── MVN helpers ────────────────────────────────────────────────────────────────

def _ar1_sigma(D, rho):
    idx = np.arange(D)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _cond_mean_j(z, j, d, rho):
    """E[X_j | X_d=z] = rho^|j-d| * z"""
    return rho ** abs(j - d) * z


def _cond_var_j(j, d, rho):
    """Var(X_j | X_d=z) = 1 - rho^(2|j-d|)"""
    return 1.0 - rho ** (2 * abs(j - d))


def _cond_mean2_j(z, j, d, rho):
    """E[X_j^2 | X_d=z] = E[X_j|z]^2 + Var(X_j|z)"""
    return _cond_mean_j(z, j, d, rho) ** 2 + _cond_var_j(j, d, rho)


def _cumtrap(integrand, grid):
    ale = np.zeros(len(grid))
    dx = np.diff(grid)
    for i in range(1, len(grid)):
        ale[i] = ale[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * dx[i - 1]
    return ale - ale.mean()


# ── g1: Bilinear  f = x1*x2   D=2 ────────────────────────────────────────────

def _g1_f(X):
    X = np.atleast_2d(X)
    return X[:, 0] * X[:, 1]


def _g1_ale(d, rho, grid):
    g = np.asarray(grid, dtype=float)
    other = 1 - d
    return _cumtrap(_cond_mean_j(g, other, d, rho), g)


def _make_g1():
    return {"f": _g1_f, "ale_fns": {d: _g1_ale for d in range(2)},
            "D": 2, "name": "Bilinear (D=2)", "description": "f = x1*x2"}


# ── g2: Squared sum  f = (x1+x2+x3+x4)^2   D=4 ───────────────────────────────

def _g2_f(X):
    X = np.atleast_2d(X)
    return X.sum(axis=1) ** 2


def _g2_ale(d, rho, grid):
    g = np.asarray(grid, dtype=float)
    D = 4
    s = g.copy()
    for j in range(D):
        if j != d:
            s = s + _cond_mean_j(g, j, d, rho)
    return _cumtrap(2 * s, g)


def _make_g2():
    return {"f": _g2_f, "ale_fns": {d: _g2_ale for d in range(4)},
            "D": 4, "name": "Squared sum (D=4)", "description": "f = (x1+x2+x3+x4)^2"}


# ── g3: Exponential sum  f = exp((x1+x2+x3+x4)/4)   D=4 ──────────────────────

def _g3_f(X):
    X = np.atleast_2d(X)
    return np.exp(X.sum(axis=1) / 4.0)


def _g3_ale(d, rho, grid):
    g = np.asarray(grid, dtype=float)
    D = 4
    mz = g.copy()
    for j in range(D):
        if j != d:
            mz = mz + _cond_mean_j(g, j, d, rho)
    Sigma = _ar1_sigma(D, rho)
    Sigma_cond = Sigma - np.outer(Sigma[:, d], Sigma[:, d]) / Sigma[d, d]
    v = float(Sigma_cond.sum())
    return _cumtrap(np.exp(mz / 4.0 + v / 32.0) / 4.0, g)


def _make_g3():
    return {"f": _g3_f, "ale_fns": {d: _g3_ale for d in range(4)},
            "D": 4, "name": "Exponential sum (D=4)",
            "description": "f = exp((x1+x2+x3+x4)/4)"}


# ── g4: Ishigami  f = sin(x1) + 7*sin^2(x2) + 0.1*x3^4*sin(x1)   D=3 ────────

def _g4_f(X):
    X = np.atleast_2d(X)
    return np.sin(X[:, 0]) + 7 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])


def _g4_ale0(d, rho, grid):
    g = np.asarray(grid, dtype=float)
    m3 = _cond_mean_j(g, 2, 0, rho)
    v3 = _cond_var_j(2, 0, rho)
    E4 = m3 ** 4 + 6 * m3 ** 2 * v3 + 3 * v3 ** 2
    return _cumtrap(np.cos(g) * (1 + 0.1 * E4), g)


def _g4_ale1(d, rho, grid):
    g = np.asarray(grid, dtype=float)
    return _cumtrap(7 * np.sin(2 * g), g)


def _g4_ale2(d, rho, grid):
    g = np.asarray(grid, dtype=float)
    m1 = _cond_mean_j(g, 0, 2, rho)
    v1 = _cond_var_j(0, 2, rho)
    return _cumtrap(0.4 * g ** 3 * np.sin(m1) * np.exp(-v1 / 2), g)


def _make_g4():
    return {"f": _g4_f, "ale_fns": {0: _g4_ale0, 1: _g4_ale1, 2: _g4_ale2},
            "D": 3, "name": "Ishigami (D=3)",
            "description": "f=sin(x1)+7sin^2(x2)+0.1*x3^4*sin(x1)"}


# ── g5: Quadratic form  f = x^T A x   D=4 ────────────────────────────────────

_rng_g5 = np.random.default_rng(42)
_M = _rng_g5.standard_normal((4, 4))
_A_G5 = (_M @ _M.T) / 4.0


def _g5_f(X):
    X = np.atleast_2d(X)
    return np.einsum('ni,ij,nj->n', X, _A_G5, X)


def _g5_ale(d, rho, grid):
    g = np.asarray(grid, dtype=float)
    D = 4
    Ad = _A_G5[d, :]
    s = Ad[d] * g
    for j in range(D):
        if j != d:
            s = s + Ad[j] * _cond_mean_j(g, j, d, rho)
    return _cumtrap(2 * s, g)


def _make_g5():
    return {"f": _g5_f, "ale_fns": {d: _g5_ale for d in range(4)},
            "D": 4, "name": "Quadratic form (D=4)",
            "description": "f = x^T A x, random PD A (seed=42)"}


# ── g6: Cyclic polynomial  D=4 ────────────────────────────────────────────────

def _g6_f(X):
    X = np.atleast_2d(X)
    return (X[:, 0] ** 2 * X[:, 1] + X[:, 1] ** 2 * X[:, 2]
            + X[:, 2] ** 2 * X[:, 3] + X[:, 3] ** 2 * X[:, 0])


def _g6_ale(d, rho, grid):
    g = np.asarray(grid, dtype=float)
    D = 4
    next_d = (d + 1) % D
    prev_d = (d - 1) % D
    Em_next = _cond_mean_j(g, next_d, d, rho)
    Em2_prev = _cond_mean2_j(g, prev_d, d, rho)
    return _cumtrap(2 * g * Em_next + Em2_prev, g)


def _make_g6():
    return {"f": _g6_f, "ale_fns": {d: _g6_ale for d in range(4)},
            "D": 4, "name": "Cyclic polynomial (D=4)",
            "description": "f=x1^2*x2+x2^2*x3+x3^2*x4+x4^2*x1"}


# ── g7: Detpep  D=8 ───────────────────────────────────────────────────────────

def _g7_f(X):
    X = np.atleast_2d(X)
    out = np.zeros(X.shape[0])
    for i in range(1, 9):
        out += (X[:, :i].sum(axis=1) - i / 2) ** 2
    return out


def _g7_ale(d, rho, grid):
    g = np.asarray(grid, dtype=float)
    D = 8
    integrand = np.zeros_like(g)
    for i in range(d + 1, D + 1):
        ESi = g.copy()
        for j in range(i):
            if j != d:
                ESi = ESi + _cond_mean_j(g, j, d, rho)
        integrand += 2 * (ESi - i / 2)
    return _cumtrap(integrand, g)


def _make_g7():
    return {"f": _g7_f, "ale_fns": {d: _g7_ale for d in range(8)},
            "D": 8, "name": "Detpep (D=8)",
            "description": "f=sum_i(S_i-i/2)^2, S_i=sum_{j<=i} x_j"}


# ── Registry ──────────────────────────────────────────────────────────────────

ENVS = {
    "g1": _make_g1(),
    "g2": _make_g2(),
    "g3": _make_g3(),
    "g4": _make_g4(),
    "g5": _make_g5(),
    "g6": _make_g6(),
    "g7": _make_g7(),
}


def true_effect_on_grid(env, d, grid, rho=0.0):
    """Analytical ALE main effect of variable d on grid, grid-centred."""
    grid = np.asarray(grid, dtype=float)
    effect = env["ale_fns"][d](d, rho, grid)
    return effect - effect.mean()


def sample(env, n, rho=0.0, noise_frac=0.1, seed=None):
    """Draw n samples from X ~ N(0, Sigma_AR1(rho)) and evaluate f(X) + noise."""
    rng = np.random.default_rng(seed)
    D = env["D"]
    Sigma = _ar1_sigma(D, rho)
    X = rng.multivariate_normal(np.zeros(D), Sigma, size=n)
    f_vals = env["f"](X)
    sigma = noise_frac * f_vals.std()
    y = f_vals + rng.normal(0, max(sigma, 1e-8), n)
    return X, y
