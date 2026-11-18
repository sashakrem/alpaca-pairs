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


def synthetic_universe(n_noise=12, n_pairs=3, n_days=1500, seed=0):
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
        y = rng.uniform(-0.5, 0.5) + beta * x + spread
        cols[f"PX{j}"], cols[f"PY{j}"] = x, y
        planted.append({"pair": (f"PY{j}", f"PX{j}"), "beta": beta,
                        "half_life": np.log(2) / theta})
    return pd.DataFrame(np.exp(pd.DataFrame(cols, index=dates))), planted
