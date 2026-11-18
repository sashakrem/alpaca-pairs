"""Ornstein-Uhlenbeck process: estimation and simulation.

The spread X_t of a cointegrated pair is modeled as

    dX_t = theta * (mu - X_t) dt + sigma dW_t

theta : speed of mean reversion (1/days)
mu    : long-run mean of the spread
sigma : instantaneous volatility

Exact solution over a step dt:

    X_{t+dt} = mu + (X_t - mu) e^{-theta dt} + eps,
    eps ~ N(0, sigma^2 (1 - e^{-2 theta dt}) / (2 theta))

so the sampled process is exactly an AR(1):  X_{t+1} = a + b X_t + eps,
with b = e^{-theta dt}, a = mu (1 - b). Fitting the AR(1) by OLS and inverting
these relations gives the (conditional) maximum-likelihood estimates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OUParams:
    theta: float      # mean-reversion speed (per unit dt)
    mu: float         # long-run mean
    sigma: float      # diffusion coefficient
    dt: float = 1.0

    @property
    def half_life(self) -> float:
        """Expected time for a deviation from mu to decay by half: ln2 / theta."""
        return np.log(2) / self.theta

    @property
    def eq_std(self) -> float:
        """Std. dev. of the stationary distribution: sigma / sqrt(2 theta)."""
        return self.sigma / np.sqrt(2 * self.theta)

    def zscore(self, x):
        """How many stationary std devs the spread is from its mean."""
        return (np.asarray(x) - self.mu) / self.eq_std


def fit_ou(x, dt: float = 1.0) -> OUParams | None:
    """Fit OU parameters via the exact AR(1) discretization.

    Returns None if the series shows no mean reversion (b outside (0, 1)).
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        return None
    x0, x1 = x[:-1], x[1:]
    b, a = np.polyfit(x0, x1, 1)
    if not (0.0 < b < 1.0):
        return None
    resid = x1 - (a + b * x0)
    s2 = resid.var(ddof=2)
    theta = -np.log(b) / dt
    mu = a / (1.0 - b)
    sigma = np.sqrt(s2 * 2.0 * theta / (1.0 - b**2))
    return OUParams(theta=theta, mu=mu, sigma=sigma, dt=dt)


def simulate_ou(theta, mu, sigma, n, x0=None, dt=1.0, method="exact", seed=None):
    """Simulate n steps of an OU process.

    method="exact": samples from the true transition density (no discretization error).
    method="euler": Euler-Maruyama, X += theta(mu - X)dt + sigma sqrt(dt) Z.
                    Biased for large theta*dt and unstable if theta*dt > 2.
    """
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu if x0 is None else x0
    z = rng.standard_normal(n - 1)
    if method == "exact":
        b = np.exp(-theta * dt)
        s = sigma * np.sqrt((1 - b**2) / (2 * theta))
        for t in range(n - 1):
            x[t + 1] = mu + (x[t] - mu) * b + s * z[t]
    elif method == "euler":
        for t in range(n - 1):
            x[t + 1] = x[t] + theta * (mu - x[t]) * dt + sigma * np.sqrt(dt) * z[t]
    else:
        raise ValueError(f"unknown method {method!r}")
    return x
