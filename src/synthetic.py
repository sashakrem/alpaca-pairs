"""Synthetic market with known ground truth, for validating the screener.

Independent stocks follow geometric random walks (never cointegrated).
Planted pairs satisfy log(Y) = c + beta*log(X) + OU spread (cointegrated by
construction). A good screener should find the planted pairs and almost
nothing else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ou import simulate_ou


def synthetic_universe(n_noise=12, n_pairs=3, n_days=1500, seed=0, beta_drift=0.0):
    """beta_drift: each planted hedge ratio moves linearly from beta to
    beta * (1 + beta_drift) over the sample (0 = constant, as in stage 1).

    Drift is applied to *returns*: dy_t = beta_t dx_t + d(spread)_t. Writing it in
    levels instead (y_t = beta_t x_t) would make y drift by d(beta) * x_t ~ d(beta) * 4
    regardless of x: two stocks that permanently diverge, which no hedge can trade."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days)

    def random_walk():
        return np.log(rng.uniform(20, 200)) + np.cumsum(rng.normal(0.0003, 0.015, n_days))

    cols, planted = {}, []
    for i in range(n_noise):
        cols[f"RW{i:02d}"] = random_walk()
    for j in range(n_pairs):
        x = random_walk()
        beta = rng.uniform(0.6, 1.4)
        theta = rng.uniform(0.04, 0.15)  # half-lives ~5-17 days
        spread = simulate_ou(theta, 0.0, 0.02, n_days, seed=int(rng.integers(1e9)))
        beta_path = beta * np.linspace(1.0, 1.0 + beta_drift, n_days)
        hedged = beta * x[0] + np.concatenate([[0.0], np.cumsum(beta_path[1:] * np.diff(x))])
        y = rng.uniform(-0.5, 0.5) + hedged + spread
        cols[f"PX{j}"], cols[f"PY{j}"] = x, y
        planted.append({"pair": (f"PY{j}", f"PX{j}"), "beta": beta, "beta_end": beta * (1 + beta_drift),
                        "half_life": np.log(2) / theta})
    return pd.DataFrame(np.exp(pd.DataFrame(cols, index=dates))), planted
