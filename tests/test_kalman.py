import warnings

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller

from src.kalman import KalmanParams, fit_kalman, kalman_filter
from src.ou import simulate_ou

N = 1250


def _random_walk(rng):
    return np.cumsum(rng.normal(0, 0.015, N)) + 4


def _portfolio_check(y, x):
    """What the screen tests: OU half-life and ADF p-value of the tradeable portfolio."""
    from src.ou import fit_ou
    from src.screen import portfolio_spread

    _, kf = fit_kalman(y, x)
    v = portfolio_spread(y, x, kf.beta)[20:]
    ou = fit_ou(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return (ou.half_life if ou else np.inf), adfuller(v)[1]


def _reference_filter(y, x, p, s0, P0):
    """Plain 3x3 matrix Kalman filter, written straight from the module docstring."""
    F = np.array([[1, 0, 0], [0, p.phi, 0], [0, 1, 0]], dtype=float)
    Q = np.diag([p.q, p.s2, 0.0])
    s, P, out = s0.copy(), P0.copy(), []
    for dy, dx in zip(np.diff(y), np.diff(x)):
        s, P = F @ s, F @ P @ F.T + Q
        H = np.array([dx, 1.0, -1.0])
        S = H @ P @ H + p.r
        K = P @ H / S
        s = s + K * (dy - H @ s)
        P = (np.eye(3) - np.outer(K, H)) @ P
        out.append(s.copy())
    return np.array(out), P


def test_compiled_filter_matches_matrix_equations():
    rng = np.random.default_rng(0)
    x = _random_walk(rng)
    y = 0.3 + 0.9 * x + simulate_ou(0.1, 0, 0.01, N, seed=1)
    p = KalmanParams(q=1e-6, phi=0.9, s2=1e-4, r=1e-5)
    s0 = np.array([1.0, 0.0, 0.0])
    P0 = np.diag([0.25, 1e-3, 1e-3])
    kf = kalman_filter(y, x, p, s0=s0, P0=P0)
    ref, Pref = _reference_filter(y, x, p, s0, P0)
    np.testing.assert_allclose(kf.beta[1:], ref[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(kf.spread[1:], ref[:, 1], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(kf.P, Pref, rtol=1e-7, atol=1e-14)


def test_recovers_ou_half_life_with_constant_beta():
    x = _random_walk(np.random.default_rng(0))
    y = 0.2 + 1.1 * x + simulate_ou(0.1, 0, 0.01, N, seed=6)
    p, kf = fit_kalman(y, x)
    assert p.half_life == pytest.approx(np.log(2) / 0.1, rel=0.2)
    err = np.abs(kf.beta[100:] - 1.1)                  # beta is constant; the filter wobbles
    assert np.median(err) < 0.04 and err.max() < 0.1   # only by estimation noise


def test_tracks_drifting_beta_that_breaks_static_regression():
    """Hedge ratio drifts 0.8 -> 1.6 in returns: dy = beta_t dx + d(OU)."""
    from src.ou import fit_ou
    from src.screen import portfolio_spread

    rng = np.random.default_rng(1)
    x = _random_walk(rng)
    beta_true = np.linspace(0.8, 1.6, N)
    y = 0.2 + 0.8 * x[0] + np.concatenate([[0], np.cumsum(beta_true[1:] * np.diff(x))]) \
        + simulate_ou(0.1, 0, 0.01, N, seed=51)
    true_hl = np.log(2) / 0.1
    # Static hedge: the leftover sum((beta_t - b) dx) contaminates the spread.
    b, a = np.polyfit(x, y, 1)
    ou_static = fit_ou(y - b * x - a)
    static_err = abs((ou_static.half_life if ou_static else np.inf) - true_hl)
    # Kalman follows beta; the tradeable portfolio mean-reverts at the true speed.
    _, kf = fit_kalman(y, x)
    assert np.median(np.abs(kf.beta[200:] - beta_true[200:])) < 0.1
    hl = fit_ou(portfolio_spread(y, x, kf.beta)[20:]).half_life
    assert hl == pytest.approx(true_hl, rel=0.35)
    assert abs(hl - true_hl) < static_err


def test_random_walks_are_rejected():
    for seed in range(100, 104):
        g = np.random.default_rng(seed)
        hl, adf_p = _portfolio_check(_random_walk(g), _random_walk(g))
        assert hl > 60 and adf_p > 0.05, (seed, hl, adf_p)


def test_textbook_random_walk_intercept_model_finds_fake_pairs():
    """The common setup: intercept AND beta follow random walks, trade the forecast error.
    With an adaptive intercept, the forecast error of two INDEPENDENT random walks is
    stationary, so this model would 'find' pairs in pure noise."""
    g = np.random.default_rng(7)
    y, x = _random_walk(g), _random_walk(g)
    r = np.var(np.diff(y))
    q = r                          # adaptivity ratio q/r ~ 1, what MLE picks for such data
    s, P, err = np.array([0.0, 1.0]), np.eye(2), []
    for t in range(N):
        P = P + q * np.eye(2)
        H = np.array([1.0, x[t]])
        e = y[t] - H @ s
        S = H @ P @ H + r
        K = P @ H / S
        s, P = s + K * e, P - np.outer(K, H @ P)
        err.append(e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert adfuller(err[20:])[1] < 0.01          # "stationary" -> fake pair
    hl, adf_p = _portfolio_check(y, x)
    assert hl > 60 and adf_p > 0.05                  # this module's approach isn't fooled


def test_portfolio_spread_rejects_leg_of_a_real_pair_vs_noise():
    """y is one leg of a real pair, so it contains a genuine OU component; z is unrelated.
    Paired with z, the tradeable portfolio must not look mean-reverting."""
    g = np.random.default_rng(12)
    x, z = _random_walk(g), _random_walk(g)
    y = 0.1 + 1.0 * x + simulate_ou(0.15, 0, 0.01, N, seed=13)
    hl_real, p_real = _portfolio_check(y, x)
    hl_fake, p_fake = _portfolio_check(y, z)
    assert hl_real < 10 and p_real < 0.001
    assert hl_fake > 60 and p_fake > 0.1
