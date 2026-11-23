"""Offline tests: no network, no API keys."""
import numpy as np
import pandas as pd
import pytest

from bot.backtest import run_backtest
from bot.broker import plan_rebalance
from bot.strategy import CASH, RISKY, month_end_prices, momentum_rotation, target_weights

UNIVERSE = RISKY + [CASH]


def fake_prices(days=600, seed=0, drift=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=days)
    mu = drift if drift is not None else {s: 0.0002 for s in UNIVERSE}
    data = {s: 100 * np.exp(np.cumsum(rng.normal(mu[s], 0.01, days))) for s in UNIVERSE}
    return pd.DataFrame(data, index=idx)


def test_weights_sum_to_one_and_pick_strongest():
    drift = {s: 0.0 for s in UNIVERSE}
    drift.update({"GLD": 0.005, "QQQ": 0.004, "EEM": 0.003, CASH: 0.00005})
    w = momentum_rotation(month_end_prices(fake_prices(drift=drift)))
    w = w[w.sum(axis=1) > 0]
    assert np.allclose(w.sum(axis=1), 1.0)
    last = w.iloc[-1]
    assert set(last[last > 0].index) == {"GLD", "QQQ", "EEM"}


def test_everything_falling_goes_to_cash():
    drift = {s: -0.002 for s in RISKY} | {CASH: 0.0001}
    w = momentum_rotation(month_end_prices(fake_prices(drift=drift)))
    assert w.iloc[-1][CASH] == pytest.approx(1.0)


def test_no_look_ahead():
    """Changing prices after a date must not change that date's weights."""
    p = fake_prices()
    cut = p.index[400]
    p2 = p.copy()
    p2.loc[p2.index > cut] *= np.linspace(0.5, 3, (p2.index > cut).sum())[:, None]
    w1 = momentum_rotation(month_end_prices(p))
    w2 = momentum_rotation(month_end_prices(p2))
    mask = w1.index <= cut
    pd.testing.assert_frame_equal(w1[mask], w2[mask])


def test_incomplete_month_is_ignored():
    p = fake_prices()
    mid_month = pd.Timestamp("2017-03-15")
    signal_date, _ = target_weights("momentum", p[p.index <= mid_month], today=mid_month)
    assert signal_date.to_period("M") == pd.Period("2017-02")


def test_backtest_buy_and_hold_matches_price_ratio():
    p = fake_prices()
    bt = run_backtest(p, p, "spy", capital=100_000, cost_bps=0)
    first = bt.index[0]
    expected = 100_000 * p["SPY"].iloc[-1] / p.loc[first, "SPY"]
    assert bt["equity"].iloc[-1] == pytest.approx(expected)


def test_backtest_charges_costs():
    p = fake_prices()
    free = run_backtest(p, p, "momentum", cost_bps=0)["equity"].iloc[-1]
    paid = run_backtest(p, p, "momentum", cost_bps=10)["equity"].iloc[-1]
    assert paid < free


def test_plan_sells_first_and_exits_dropped_positions():
    holdings = {"SPY": 50_000, "TLT": 50_000}
    targets = {"SPY": 1 / 3, "GLD": 1 / 3, "QQQ": 1 / 3}
    plan = plan_rebalance(100_000, holdings, targets, UNIVERSE)
    sides = [o.side for o in plan]
    assert sides == sorted(sides, reverse=True)          # all sells before buys
    tlt = next(o for o in plan if o.symbol == "TLT")
    assert tlt.full_exit and tlt.notional == 50_000
    bought = sum(o.notional for o in plan if o.side == "buy")
    sold = sum(o.notional for o in plan if o.side == "sell")
    assert bought - sold <= 0                              # never spends more than it frees (1% buffer)


def test_plan_skips_tiny_trades_and_ignores_other_symbols():
    holdings = {"SPY": 99_000 * (1 / 3) + 10}
    targets = {"SPY": 1 / 3}
    plan = plan_rebalance(100_000, holdings | {"AAPL": 5_000}, targets, UNIVERSE)
    assert plan == []


def test_execute_against_fake_alpaca_client():
    """Runs the real order code against a stand-in client, checking the alpaca-py
    request objects it builds and the sell -> wait -> buy order."""
    from types import SimpleNamespace as NS
    from alpaca.trading.enums import OrderStatus

    from bot.broker import Order, PaperBroker

    calls = []

    class FakeClient:
        def get_all_positions(self):
            return [NS(symbol="TLT", qty="123.456", market_value="10000", current_price="81")]

        def submit_order(self, req):
            calls.append(req)
            return NS(id=len(calls), symbol=req.symbol)

        def get_order_by_id(self, oid):
            return NS(id=oid, symbol=calls[oid - 1].symbol, status=OrderStatus.FILLED)

        def get_account(self):
            return NS(cash="5000")

        def get_asset(self, sym):
            return NS(fractionable=sym != "DBC")

    b = PaperBroker.__new__(PaperBroker)
    b.client = FakeClient()
    plan = [Order("TLT", "sell", 10_000, full_exit=True),
            Order("GLD", "buy", 6_000), Order("DBC", "buy", 4_000)]
    b.execute(plan, "2026-10", prices={"DBC": 22.0}, log=lambda *_: None)

    sell, gld, dbc = calls
    assert sell.side.value == "sell" and float(sell.qty) == pytest.approx(123.456)
    assert all(c.client_order_id.startswith("bot-2026-10-") for c in calls)
    # only $5,000 cash available -> buys scaled by half
    assert float(gld.notional) == pytest.approx(3_000)
    assert float(dbc.qty) == 90 and dbc.notional is None   # $2,000 / $22, whole shares
