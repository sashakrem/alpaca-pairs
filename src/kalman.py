"""Kalman filter: time-varying hedge ratio with an Ornstein-Uhlenbeck spread.

Two traps this module avoids
----------------------------
1. The textbook "Kalman pairs trading" model lets an intercept follow a random
   walk and trades the forecast error. Fitted by maximum likelihood, the
   intercept absorbs the spread, so the errors look stationary for ANY two
   stocks, even independent random walks (tests/test_kalman.py shows this).

2. Writing the model in log-price LEVELS, y_t = beta_t x_t + m_t, is subtler but
   similar: log prices sit around 4, so small moves in beta_t shift y by
   4 * d(beta) and act like a free intercept. Also, the beta that best fits
   levels is not the beta that hedges returns, so it lags real drift.

So the model is written in RETURNS, with the spread as an explicit OU state.

Model (state space)
-------------------
Hidden state s_t = [beta_t, m_t, m_{t-1}]':

    beta_t = beta_{t-1} + w_t,                 w_t   ~ N(0, q)    hedge ratio drifts slowly
    m_t    = phi m_{t-1} + eta_t,              eta_t ~ N(0, s2)   spread: discretized OU around 0

Observed daily log returns dy_t, dx_t:

    dy_t = beta_t dx_t + (m_t - m_{t-1}) + v_t,   v_t ~ N(0, r)   r: tick / bid-ask noise

phi = e^{-theta dt} links this to the OU SDE dm = -theta m dt + sigma dW. The
half-life is ln2 / theta = -ln2 / ln(phi). The stationary std of m is
sqrt(s2 / (1 - phi^2)). The spread's long-run level drops out when
differencing; m measures the deviation from it.

Matrix form:  s_t = F s_{t-1} + noise,  dy_t = H_t s_t + v_t,

    F = [[1, 0,   0],      Q = diag(q, s2, 0),      H_t = [dx_t, 1, -1]
         [0, phi, 0],
         [0, 1,   0]]

Each day:
    Predict:  s- = F s,                   P- = F P F' + Q
    Innovate: e = dy_t - H_t s-,          S  = H_t P- H_t' + r
    Update:   K = P- H_t' / S,            s  = s- + K e,       P = (I - K H_t) P-

The parameters (q, phi, s2, r) are fitted by maximizing the prediction-error
log-likelihood  -1/2 sum[ log(2 pi S_t) + e_t^2 / S_t ].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import minimize


@dataclass(frozen=True)
class KalmanParams:
    q: float      # hedge-ratio random-walk variance per day
    phi: float    # spread AR(1) coefficient = e^{-theta}
    s2: float     # spread innovation variance
    r: float      # observation noise variance

    @property
    def theta(self) -> float:
        return -np.log(self.phi)

    @property
    def half_life(self) -> float:
        return np.log(2) / self.theta if self.phi < 1 else np.inf

    @property
    def eq_std(self) -> float:
        return np.sqrt(self.s2 / (1 - self.phi**2))


@dataclass
class KalmanResult:
    beta: np.ndarray     # beta_{t|t}: hedge ratio estimated with data up to day t
    spread: np.ndarray   # m_{t|t}: spread deviation estimated with data up to day t
    loglik: float
    state: np.ndarray    # final s_{T|T} (to continue filtering on new data)
    P: np.ndarray        # final P_{T|T}

    def zscore(self, p: KalmanParams) -> np.ndarray:
        return self.spread / p.eq_std


@njit(cache=True)
def _filter_core(dy, dx, q, phi, s2, r, s, P, burn_in):
    """Filter loop, compiled by numba. s (3,) and P (3x3) are updated in place."""
    n = len(dy)
    beta = np.empty(n)
    spread = np.empty(n)
    F = np.array([[1.0, 0.0, 0.0], [0.0, phi, 0.0], [0.0, 1.0, 0.0]])
    Q = np.diag(np.array([q, s2, 0.0]))
    H = np.zeros(3)
    ll = 0.0
    for t in range(n):
        s[:] = F @ s                              # predict
        P[:, :] = F @ P @ F.T + Q
        H[0] = dx[t]
        H[1] = 1.0
        H[2] = -1.0
        PH = P @ H
        S = H @ PH + r
        e = dy[t] - H @ s                         # innovation
        K = PH / S                                # gain
        s += K * e                                # update
        P -= np.outer(K, PH)
        P[:, :] = 0.5 * (P + P.T)                 # keep symmetric against round-off
        beta[t] = s[0]
        spread[t] = s[1]
        if t >= burn_in:
            ll += -0.5 * (np.log(2.0 * np.pi * S) + e * e / S)
    return beta, spread, ll


def kalman_filter(y, x, p: KalmanParams, s0=None, P0=None, burn_in: int = 20) -> KalmanResult:
    """Run the filter over log-price arrays y, x (length n) with fixed parameters.

    Output arrays have length n; day 0 holds the starting state. Pass s0/P0 from a
    previous run's .state/.P to continue it on new data.
    """
    dy = np.diff(np.asarray(y, dtype=float))
    dx = np.diff(np.asarray(x, dtype=float))
    m_var = p.s2 / (1 - p.phi**2)
    if s0 is None:
        k = max(burn_in, 2)
        b0 = float(np.dot(dx[:k], dy[:k]) / np.dot(dx[:k], dx[:k]))
        s0 = np.array([b0, 0.0, 0.0])
    if P0 is None:  # vague on beta; stationary prior on the spread
        P0 = np.array([[0.25, 0.0, 0.0],
                       [0.0, m_var, p.phi * m_var],
                       [0.0, p.phi * m_var, m_var]])
    s, P = np.array(s0, dtype=float), np.array(P0, dtype=float)
    beta, spread, ll = _filter_core(dy, dx, p.q, p.phi, p.s2, p.r, s, P, burn_in)
    beta = np.concatenate([[s0[0]], beta])
    spread = np.concatenate([[s0[1]], spread])
    return KalmanResult(beta, spread, ll, s, P)


def _unpack(z, scale):
    """Unconstrained vector -> valid parameters (positivity, 0 < phi < 1)."""
    lq, zphi, ls2, lr = z
    return KalmanParams(q=np.exp(lq), phi=1 / (1 + np.exp(-zphi)),
                        s2=np.exp(ls2) * scale, r=np.exp(lr) * scale)


def fit_kalman(y, x, burn_in: int = 20) -> tuple[KalmanParams, KalmanResult]:
    """Maximum-likelihood estimates of the parameters, plus the filter run with them."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    dy, dx = np.diff(y), np.diff(x)
    b = np.dot(dx, dy) / np.dot(dx, dx)
    scale = np.var(dy - b * dx)   # variances measured relative to the hedged daily return variance

    def nll(z):
        return -kalman_filter(y, x, _unpack(z, scale), burn_in=burn_in).loglik

    # The likelihood can have several local optima (e.g. "fast-reverting spread +
    # drifting beta" vs "slow spread that absorbs the drift"). Start from a grid
    # over (q, phi) and keep the best.
    best = None
    for phi0 in (0.5, 0.9, 0.99):
        for lq0 in (np.log(1e-8), np.log(1e-6), np.log(1e-4)):
            z0 = [lq0, np.log(phi0 / (1 - phi0)), np.log(0.5), np.log(0.2)]
            res = minimize(nll, z0, method="L-BFGS-B",
                           bounds=[(np.log(1e-12), np.log(1e-2)), (-5, 12), (-12, 4), (-12, 4)])
            if best is None or res.fun < best.fun:
                best = res
    p = _unpack(best.x, scale)
    return p, kalman_filter(y, x, p, burn_in=burn_in)
