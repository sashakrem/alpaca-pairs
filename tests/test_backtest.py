import numpy as np
import pandas as pd
import pytest

from src.backtest import Rules, performance, simulate_pair, walk_forward
from src.ou import simulate_ou
from src.synthetic import synthetic_universe


def _pair_frame(spread, beta=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = len(spread)
    lx = 4 + np.cumsum(rng.normal(0, 0.01, n))
    ly = beta * lx + spread
    return pd.DataFrame({"Y": ly, "X": lx}, index=pd.bdate_range("2020-01-01", periods=n))


def test_hand_computed_trade():
    # Spread (= z, since mu=0, eq_std=1): flat, dips to -2.5 (entry), recovers to 0 (exit).
    s = np.array([0, 0, -2.5, -2.0, -1.0, 0.0, 0, 0, 0, 0], float)
    logp = _pair_frame(s * 0.01)          # z = spread / 0.01
    rules = Rules(cost_bps=0, borrow_rate=0, lag=1)
    ret, trades, turn = simulate_pair(logp, "Y", "X", 0.0, 1.0, 0.0, 0.01, 10, rules)
    assert len(trades) == 1
    tr = trades[0]
    # Signal on day 2 -> executed day 3. Exit signal on day 5 (|z| < 0.5) -> executed day 6.
    assert tr["side"] == 1 and tr["entry"] == logp.index[3] and tr["exit"] == logp.index[6]
    assert tr["exit_reason"] == "target"
    # Held over days 4..6: earns half the spread change from day 3 to day 6 (w_y = 1/2)
    # in log terms; simple returns make it slightly different.
    assert tr["pnl"] == pytest.approx(0.5 * 0.02, rel=0.05)
    assert turn.sum() == 2


def test_costs_are_charged_per_side():
    s = np.array([0, 0, -2.5, -2.0, -1.0, 0.0, 0, 0, 0, 0], float)
    logp = _pair_frame(s * 0.01)
    free, *_ = simulate_pair(logp, "Y", "X", 0.0, 1.0, 0.0, 0.01, 10, Rules(cost_bps=0, borrow_rate=0))
    paid, *_ = simulate_pair(logp, "Y", "X", 0.0, 1.0, 0.0, 0.01, 10, Rules(cost_bps=10, borrow_rate=0))
    assert (free - paid).sum() == pytest.approx(2 * 10 / 1e4)


def test_no_lookahead():
    """Scrambling prices after day k must not change returns up to day k."""
    spread = simulate_ou(0.1, 0, 0.02, 300, seed=3)
    logp = _pair_frame(spread)
    rules = Rules()
    base, *_ = simulate_pair(logp, "Y", "X", 0.0, 1.0, 0.0, 0.045, 7, rules)
    k = 150
    scrambled = logp.copy()
    scrambled.iloc[k + 1:] += np.random.default_rng(9).normal(0, 0.05, (300 - k - 1, 2))
    alt, *_ = simulate_pair(scrambled, "Y", "X", 0.0, 1.0, 0.0, 0.045, 7, rules)
    np.testing.assert_allclose(base.iloc[:k + 1], alt.iloc[:k + 1])


def test_ends_flat_at_year_end():
    spread = np.full(30, -0.1)            # z = -2.2 throughout: enters and never hits target
    logp = _pair_frame(spread)
    _, trades, turn = simulate_pair(logp, "Y", "X", 0.0, 1.0, 0.0, 0.045, 100, Rules())
    assert len(trades) == 1 and trades[0]["exit_reason"] == "year end"
    assert turn.sum() == 2


def test_walk_forward_profits_on_planted_pairs_and_not_on_noise():
    prices, _ = synthetic_universe(n_noise=6, n_pairs=3, n_days=2000, seed=4)
    res = walk_forward(prices, 2022, 2026, rules=Rules(cost_bps=0, borrow_rate=0), verbose=False)
    perf = performance(res["returns"], res["trades"], res["turnover"])
    assert perf["sharpe"] > 1.0
    planted = {f"P{s}{i}" for s in "XY" for i in range(3)}
    real = res["picks"]["y"].isin(planted) & res["picks"]["x"].isin(planted)
    assert real.mean() >= 0.8   # FDR 5% per yearly screen: a stray false pick is possible

    noise, _ = synthetic_universe(n_noise=10, n_pairs=0, n_days=2000, seed=5)
    res = walk_forward(noise, 2022, 2026, verbose=False)
    assert len(res["trades"]) <= 5   # nothing real to find; at most a stray false positive
