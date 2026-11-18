import numpy as np
import pytest

from src.ou import fit_ou, simulate_ou
from src.screen import benjamini_hochberg, screen_pairs
from src.synthetic import synthetic_universe


def test_ou_fit_recovers_parameters():
    theta, mu, sigma = 0.1, 0.5, 0.03
    x = simulate_ou(theta, mu, sigma, 50_000, seed=1)
    p = fit_ou(x)
    assert p.theta == pytest.approx(theta, rel=0.05)
    assert p.mu == pytest.approx(mu, abs=0.01)
    assert p.sigma == pytest.approx(sigma, rel=0.02)
    assert p.half_life == pytest.approx(np.log(2) / theta, rel=0.05)


def test_euler_bias_grows_with_step_size():
    # Euler-Maruyama's AR coefficient is (1 - theta dt) instead of e^{-theta dt}.
    # The fitted theta is therefore biased low, worse as theta*dt grows.
    errs = []
    for theta in (0.05, 0.5, 1.0):
        x = simulate_ou(theta, 0, 1, 50_000, method="euler", seed=2)
        errs.append(abs(fit_ou(x).theta - theta) / theta)
    assert errs[0] < errs[1] < errs[2]


def test_random_walk_has_no_mean_reversion_or_long_half_life():
    rw = np.cumsum(np.random.default_rng(3).standard_normal(2000))
    p = fit_ou(rw)
    assert p is None or p.half_life > 100


def test_benjamini_hochberg_known_example():
    p = [0.01, 0.04, 0.03, 0.005, 0.5]
    reject, adj = benjamini_hochberg(p, q=0.05)
    assert reject.tolist() == [True, True, True, True, False]
    assert adj[3] == pytest.approx(0.025)


def test_screener_finds_planted_pairs_only():
    prices, planted = synthetic_universe(seed=0)
    res, s = screen_pairs(prices, "2023-01-01")
    found = set(zip(res.loc[res.verdict == "CANDIDATE", "y"], res.loc[res.verdict == "CANDIDATE", "x"]))
    truth = {p["pair"] for p in planted}
    truth |= {(b, a) for a, b in truth}
    assert all(any(f == t for t in truth) for f in found), "false positive"
    assert len(found) == len(planted)
    for p in planted:
        row = res[res.y.isin(p["pair"]) & res.x.isin(p["pair"])].iloc[0]
        if row.y == p["pair"][0]:
            assert row.beta == pytest.approx(p["beta"], rel=0.05)
