"""Strategies: turn month-end prices into target portfolio weights.

Every strategy is a function  month_end_prices (DataFrame, rows = month ends,
columns = symbols)  ->  weights (DataFrame, same index, columns = symbols, rows sum to 1).

The weights in row d use only prices up to and including d. They are traded on the
next trading day, so there is no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CASH = "BIL"  # 1-3 month T-bill ETF: where money waits when nothing is worth holding

# Ten liquid ETFs covering the major asset classes.
RISKY = [
    "SPY",  # US large caps
    "QQQ",  # US tech / Nasdaq 100
    "IWM",  # US small caps
    "EFA",  # developed markets ex-US
    "EEM",  # emerging markets
    "TLT",  # long-term Treasuries
    "IEF",  # intermediate Treasuries
    "GLD",  # gold
    "DBC",  # commodities
    "VNQ",  # US real estate
]

LOOKBACKS = (1, 3, 6, 12)  # months
TOP_K = 3


def momentum_score(me: pd.DataFrame) -> pd.DataFrame:
    """Average of the 1-, 3-, 6- and 12-month total returns.

    Blending lookbacks makes the ranking less sensitive to any single window
    (a 12-month signal alone reacts very slowly to crashes).
    """
    return sum(me / me.shift(k) - 1 for k in LOOKBACKS) / len(LOOKBACKS)


def momentum_rotation(me: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """Hold the top_k ETFs by momentum score, equal weight.

    Absolute-momentum filter: a pick only gets its slot if it beat T-bills (BIL).
    Otherwise that slot goes to BIL. In a broad sell-off every asset fails this test
    and the portfolio moves to cash, which is what limits crash losses.
    """
    score = momentum_score(me)
    w = pd.DataFrame(0.0, index=me.index, columns=RISKY + [CASH])
    for d in me.index:
        s = score.loc[d, RISKY]
        if s.isna().any() or np.isnan(score.loc[d, CASH]):
            continue  # not enough history yet
        for sym in s.nlargest(top_k).index:
            if s[sym] > score.loc[d, CASH]:
                w.loc[d, sym] = 1.0 / top_k
        w.loc[d, CASH] = 1.0 - w.loc[d, RISKY].sum()
    return w


def buy_and_hold(me: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=me.index, columns=["SPY"])
    w["SPY"] = 1.0
    return w


def sixty_forty(me: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=me.index, columns=["SPY", "IEF"])
    w["SPY"], w["IEF"] = 0.6, 0.4
    return w


STRATEGIES = {
    "momentum": (momentum_rotation, RISKY + [CASH]),
    "spy": (buy_and_hold, ["SPY"]),
    "60-40": (sixty_forty, ["SPY", "IEF"]),
}


def month_end_prices(daily: pd.DataFrame, today=None) -> pd.DataFrame:
    """Last close of each COMPLETED month.

    The current month is dropped if `today` falls inside it: a month-end signal
    must not use a half-finished month.
    """
    me = daily.groupby(daily.index.to_period("M")).tail(1)
    today = pd.Timestamp(today) if today is not None else daily.index[-1] + pd.Timedelta(days=1)
    current = today.to_period("M")
    return me[me.index.to_period("M") < current]


def target_weights(strategy: str, daily: pd.DataFrame, today=None) -> tuple[pd.Timestamp, pd.Series]:
    """Weights to hold right now: the signal from the last completed month end."""
    fn, _ = STRATEGIES[strategy]
    me = month_end_prices(daily, today)
    w = fn(me)
    w = w[w.sum(axis=1) > 0]
    if w.empty:
        raise ValueError("Not enough price history for this strategy (needs 13+ months).")
    last = w.iloc[-1]
    return w.index[-1], last[last > 0]
