import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _u(x):
    """Map x in [0,1] to u in [-1,1]."""
    return 2.0 * np.asarray(x, dtype=float) - 1.0


def _make_env(components_raw, name, description):
    """
    Build an env dict from a list of D raw component functions.

    Returns dict with keys:
        f           : callable (N, D) -> (N,)
        components  : list of D raw callables  g_d: array -> array
        D           : int
        name, description : str
    """
    D = len(components_raw)

    def f(X):
        X = np.atleast_2d(X)
        return sum(g(X[:, d]) for d, g in enumerate(components_raw))

    return {
        "f":           f,
        "components":  components_raw,
        "D":           D,
        "name":        name,
        "description": description,
    }


def true_effect_on_grid(env, d, grid):
    vals = env["components"][d](np.asarray(grid, dtype=float))
    return vals - vals.mean()


# f0  D=3
# f(x) = x1 + x2^2 + 0*x3

def _f0_components():
    return [
        lambda x: np.asarray(x, dtype=float),                         # x1
        lambda x: np.asarray(x, dtype=float) ** 2,                    # x2^2
        lambda x: np.zeros_like(np.asarray(x, dtype=float)),          # 0 (dummy)
    ]


# f1  D=2
# f(x) = sin(u1) + u2^2

def _f1_components():
    return [
        lambda x: np.sin(_u(x)),
        lambda x: _u(x) ** 2,
    ]


# f2  D=4
# f(x) = sin(10u1) + sin(u2) + (u3^3 - u3) + 1/(1+exp(-10u4))

def _f2_components():
    return [
        lambda x: np.sin(10.0 * _u(x)),
        lambda x: np.sin(_u(x)),
        lambda x: _u(x) ** 3 - _u(x),
        lambda x: 1.0 / (1.0 + np.exp(-10.0 * _u(x))),
    ]


# f3  D=4
# f(x) = 1(u1>0)*u1^2 + |u2|^0.5*sign(u2) + sin(pi*u3/2) + u4*log(|u4|+1)

def _f3_components():
    return [
        lambda x: np.where(_u(x) > 0, _u(x) ** 2, 0.0),
        lambda x: np.abs(_u(x)) ** 0.5 * np.sign(_u(x)),
        lambda x: np.sin(np.pi * _u(x) / 2.0),
        lambda x: _u(x) * np.log(np.abs(_u(x)) + 1.0),
    ]


# f4  D=4
# f(x) = 10*sin(u1) + 0.1*u2^2 + 5*exp(-u3^2) + 0.05*u4

def _f4_components():
    return [
        lambda x: 10.0 * np.sin(_u(x)),
        lambda x: 0.1 * _u(x) ** 2,
        lambda x: 5.0 * np.exp(-_u(x) ** 2),
        lambda x: 0.05 * _u(x),
    ]


# f5  D=8
# g_k(x_k) = (1/k)*sin(k*pi*u_k) + k*cos(pi*u_k/(2k)),  k = 1,...,8

def _f5_components():
    comps = []
    for k in range(1, 9):
        k_ = float(k)
        comps.append(
            lambda x, k=k_: (1.0 / k) * np.sin(k * np.pi * _u(x))
                            + k * np.cos(np.pi * _u(x) / (2.0 * k))
        )
    return comps


# f6  D=10
#  x1 : sum_{j in {1,3,7,15}} sin(j*pi*u) / j          irregular Fourier
#  x2 : sum_{j in {2,5,11}}   cos(j*pi*u) / j          irregular Fourier
#  x3 : (u^5 - 10u^3 + u*exp(-u^2)) / 9               high-degree polynomial
#  x4 : u / (1+u^2)                                    rational
#  x5 : tanh(5u)                                       steep sigmoid
#  x6 : exp(-u^2/0.25)                                 sharp Gaussian peak
#  x7 : log(u^2+1)*sin(3u)                             oscillatory x log
#  x8 : 1/(u^2+0.1)                                    heavy-tailed
#  x9 : sin(u^3 - u)                                   non-polynomial osc
#  x10: |u|^(1/3)*sign(u)                              cube-root (inf deriv at 0)

def _f6_components():
    def g1(x):
        u = _u(x)
        return sum(np.sin(j * np.pi * u) / j for j in [1, 3, 7, 15])

    def g2(x):
        u = _u(x)
        return sum(np.cos(j * np.pi * u) / j for j in [2, 5, 11])

    def g3(x):
        u = _u(x)
        return (u ** 5 - 10.0 * u ** 3 + u * np.exp(-u ** 2)) / 9.0

    def g4(x):
        u = _u(x)
        return u / (1.0 + u ** 2)

    def g5(x):
        return np.tanh(5.0 * _u(x))

    def g6(x):
        return np.exp(-_u(x) ** 2 / 0.25)

    def g7(x):
        u = _u(x)
        return np.log(u ** 2 + 1.0) * np.sin(3.0 * u)

    def g8(x):
        return 1.0 / (_u(x) ** 2 + 0.1)

    def g9(x):
        u = _u(x)
        return np.sin(u ** 3 - u)

    def g10(x):
        u = _u(x)
        return np.abs(u) ** (1.0 / 3.0) * np.sign(u)

    return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]


# ENVS registry

_BUILDERS = {
    "f0": (_f0_components,
           "Simple (D=3)",
           "x1 + x2^2 + 0*x3 — linear, quadratic, exact dummy"),
    "f1": (_f1_components,
           "Mixed frequency (D=2)",
           "sin(u1) + u2^2"),
    "f2": (_f2_components,
           "Varying smoothness (D=4)",
           "sin(10u), sin(u), u^3-u, sigmoid(10u)"),
    "f3": (_f3_components,
           "Near-discontinuity (D=4)",
           "kink at 0, sqrt-sign, sin(pi*u/2), u*log(|u|+1)"),
    "f4": (_f4_components,
           "Amplitude imbalance (D=4)",
           "10*sin, 0.1*u^2, 5*exp(-u^2), 0.05*u"),
    "f5": (_f5_components,
           "Multi-scale oscillation (D=8)",
           "(1/k)*sin(k*pi*u) + k*cos(pi*u/2k), k=1..8"),
    "f6": (_f6_components,
           "Fourier + polynomial + rational (D=10)",
           "10 distinct component types — rich high-dimensional benchmark"),
}

ENVS = {
    key: _make_env(builder(), name, desc)
    for key, (builder, name, desc) in _BUILDERS.items()
}


# Data sampler

def sample(env, n, dependence="independent", noise_frac=0.1, seed=None):
    """
    Draw n training samples (X, y) from the given environment.

    Parameters
    ----------
    env         : dict from ENVS
    n           : int, number of samples
    dependence  : "independent" | "low" | "high"
                    independent — all x_d ~ U[0,1] i.i.d.
                    low         — x_d = clip(x1 + N(0, 0.1^2), 0, 1)
                    high        — x_d = clip(x1 + N(0, 0.05^2), 0, 1)
    noise_frac  : float, noise std = noise_frac * std(f(X))
    seed        : int or None

    Returns
    -------
    X : array (n, D)
    y : array (n,)
    """
    rng = np.random.default_rng(seed)
    D   = env["D"]
    x1  = rng.uniform(0.0, 1.0, n)

    if dependence == "independent":
        X = rng.uniform(0.0, 1.0, (n, D))

    elif dependence == "low":
        X = np.column_stack(
            [x1] + [np.clip(x1 + rng.normal(0, 0.1, n), 0, 1)
                    for _ in range(D - 1)]
        )

    elif dependence == "high":
        X = np.column_stack(
            [x1] + [np.clip(x1 + rng.normal(0, 0.05, n), 0, 1)
                    for _ in range(D - 1)]
        )

    else:
        raise ValueError(
            f"dependence must be 'independent', 'low', or 'high'. Got '{dependence}'"
        )

    f_vals = env["f"](X)
    sigma  = noise_frac * f_vals.std()
    y      = f_vals + rng.normal(0, max(sigma, 1e-8), n)
    return X, y
