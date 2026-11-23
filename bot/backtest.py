"""Daily backtest that trades the way the live bot does.

- The signal is computed from the close of the last trading day of each month.
- Trades execute at the OPEN of the next trading day (the live bot runs in the morning).
- Holdings are tracked in shares, so weights drift between rebalances, as they do live.
- Every dollar traded pays `cost_bps` (spread + slippage; Alpaca charges no commission).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import STRATEGIES, month_end_prices


def run_backtest(opens: pd.DataFrame, closes: pd.DataFrame, strategy: str,
                 start=None, capital: float = 100_000.0, cost_bps: float = 5.0) -> pd.DataFrame:
    """Returns a daily DataFrame with columns equity, turnover, costs."""
    fn, symbols = STRATEGIES[strategy]
    opens, closes = opens[symbols], closes[symbols]
    weights = fn(month_end_prices(closes))
    weights = weights[weights.sum(axis=1) > 0]

    # Map each signal date to the next trading day, when it gets traded.
    days = closes.index
    trade_on = {}
    for d, w in weights.iterrows():
        i = days.searchsorted(d, side="right")
        if i < len(days) and (start is None or days[i] >= pd.Timestamp(start)):
            trade_on[days[i]] = w

    cash, shares = capital, pd.Series(0.0, index=symbols)
    rows, started = [], False
    rate = cost_bps / 1e4
    for day in days:
        turnover = cost = 0.0
        if day in trade_on:
            started = True
            px = opens.loc[day]
            equity = cash + float((shares * px).sum())
            target_value = trade_on[day] * equity
            trade_value = target_value - shares * px
            turnover = float(trade_value.abs().sum())
            cost = turnover * rate
            shares = target_value / px
            cash = equity - float(target_value.sum()) - cost
        if started:
            equity = cash + float((shares * closes.loc[day]).sum())
            rows.append((day, equity, turnover, cost))
    return pd.DataFrame(rows, columns=["date", "equity", "turnover", "costs"]).set_index("date")


def summarize(equity: pd.Series) -> dict:
    r = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    dd = equity / equity.cummax() - 1
    yearly = equity.resample("YE").last().pct_change()
    yearly.iloc[0] = equity.resample("YE").last().iloc[0] / equity.iloc[0] - 1
    return {
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
        "final": equity.iloc[-1],
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "cagr": (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1,
        "vol": r.std() * np.sqrt(252),
        "sharpe": r.mean() / r.std() * np.sqrt(252),  # vs 0, not vs T-bills
        "max_drawdown": dd.min(),
        "worst_year": yearly.min(),
        "losing_years": int((yearly < 0).sum()),
        "years": len(yearly),
        "yearly": yearly,
    }
