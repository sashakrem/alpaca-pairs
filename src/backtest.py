"""Walk-forward backtest of the OU pairs strategy.

Timeline (no look-ahead)
------------------------
For each trading year Y:
  1. Selection window = the `formation_years` years before Y.
     Screen with stage 1 (Engle-Granger + Benjamini-Hochberg), fitting on all but
     the last year and validating on the last year (out-of-sample ADF).
  2. For each surviving pair, refit the hedge ratio and the OU model on the whole
     selection window. Those numbers are frozen for year Y.
  3. Trade year Y with the frozen parameters. The signal is computed at the close
     of day t; the position changes at the close of day t+1 (one day of execution lag).

Trading rules (fixed before looking at any results)
---------------------------------------------------
z = (spread - mu) / eq_std, with spread = log Y - beta log X - alpha.
  enter long spread  (long Y, short X)  when -stop_z < z < -entry
  enter short spread (short Y, long X)  when +entry < z < +stop_z
  exit when |z| < exit_z, or |z| > stop_z (stop-loss),
       or after max_hold_mult * half-life days (time stop), or at year end.

Positions and P&L
-----------------
A unit position holds w_y = 1/(1+|beta|) dollars of Y and -beta/(1+|beta|) of X,
so the gross exposure is $1. Daily return of the pair = pos * (w_y r_y - w_x r_x).
Weights are treated as constant dollar weights (i.e. rebalanced daily); the
rebalancing trades are small and not charged.

Costs: `cost_bps` per side on gross notional traded (slippage + half spread;
Alpaca has no commission), plus `borrow_rate` per year on the short leg while open.

Pair selection: `coint` (stage-1 statistical screen, default) or `distance`
(Gatev-Goetzmann-Rouwenhorst benchmark: closest 5 pairs by price distance,
no significance test). Same trading rules and costs for both.

Capital: each year, capital is split equally across max(n_pairs, min_slots) slots,
so a year with one pair doesn't put 100% of capital on it. Idle cash earns 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ou import fit_ou
from .screen import hedge_regression, screen_pairs


@dataclass(frozen=True)
class Rules:
    entry: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_hold_mult: float = 3.0
    cost_bps: float = 5.0
    borrow_rate: float = 0.005
    lag: int = 1


def simulate_pair(logp: pd.DataFrame, y: str, x: str, alpha: float, beta: float,
                  mu: float, eq_std: float, half_life: float, rules: Rules):
    """Trade one pair over the dates in `logp` with frozen parameters.

    Returns (daily net returns per $1 of gross capital, list of trade dicts).
    """
    ly, lx = logp[y].values, logp[x].values
    n = len(ly)
    z = (ly - beta * lx - alpha - mu) / eq_std
    wy, wx = 1 / (1 + abs(beta)), beta / (1 + abs(beta))
    ry = np.concatenate([[0.0], np.expm1(np.diff(ly))])   # simple returns
    rx = np.concatenate([[0.0], np.expm1(np.diff(lx))])
    max_hold = max(1, int(round(rules.max_hold_mult * half_life))) if np.isfinite(half_life) else n + 1

    # 1) Desired position from signals (decided at each close).
    desired = np.zeros(n)
    pos, held, reason = 0, 0, None
    reasons = [None] * n
    for t in range(n):
        if pos == 0:          # entries and exits are separate branches: no same-day flip
            if -rules.stop_z < z[t] < -rules.entry:          # don't enter beyond the stop
                pos, held = 1, 0
            elif rules.entry < z[t] < rules.stop_z:
                pos, held = -1, 0
        else:
            held += 1
            if abs(z[t]) < rules.exit_z:
                pos, reason = 0, "target"
            elif abs(z[t]) > rules.stop_z:
                pos, reason = 0, "stop"
            elif held >= max_hold:
                pos, reason = 0, "time"
            if pos == 0:
                reasons[t] = reason
        desired[t] = pos
    desired[-1 - rules.lag:] = 0          # flat by year end (accounting for the lag)

    # 2) Actual position: executed `lag` days later.
    actual = np.concatenate([np.zeros(rules.lag), desired[:-rules.lag]]) if rules.lag else desired
    actual[-1] = 0

    # 3) P&L: position held from close t-1 earns day t's return.
    held_pos = np.concatenate([[0.0], actual[:-1]])
    gross = held_pos * (wy * ry - wx * rx)
    turnover = np.abs(np.diff(np.concatenate([[0.0], actual])))       # in units of $1 gross
    short_frac = np.where(held_pos > 0, abs(wx), np.where(held_pos < 0, wy, 0.0))
    net = gross - turnover * rules.cost_bps / 1e4 - short_frac * rules.borrow_rate / 252
    dates = logp.index

    # Trade log. Positions always pass through 0 (no direct flips), so a trade is
    # one run of nonzero `actual`. Its exit was decided `lag` days before it executed.
    trades, start = [], None
    for t in range(n):
        prev = actual[t - 1] if t else 0.0
        if prev == 0 and actual[t] != 0:
            start = t
        elif prev != 0 and actual[t] == 0:
            k = t - rules.lag
            trades.append({"y": y, "x": x, "side": int(prev), "entry": dates[start], "exit": dates[t],
                           "days": t - start,
                           "pnl": float(np.prod(1 + net[start:t + 1]) - 1),   # includes entry & exit costs
                           "exit_reason": reasons[k] if 0 <= k < n and reasons[k] else "year end"})
    return pd.Series(net, index=dates), trades, turnover


def select_cointegration(prices, year, formation_years=3, fdr=0.05):
    """Stage-1 screen on the `formation_years` before `year` (last year = validation),
    then refit hedge ratio + OU on the whole window. Returns a list of pair dicts."""
    window = prices[(prices.index >= f"{year - formation_years}-01-01") & (prices.index < f"{year}-01-01")]
    if len(window) < 250 * formation_years * 0.9:
        return []
    res, _ = screen_pairs(window, f"{year - 1}-01-01", fdr=fdr)
    if res.empty:
        return []
    lw = np.log(window)
    out = []
    for r in res[res.verdict == "CANDIDATE"].itertuples():
        alpha, beta = hedge_regression(lw[r.y].values, lw[r.x].values)
        ou = fit_ou(lw[r.y].values - beta * lw[r.x].values - alpha)
        if ou is not None:
            out.append({"y": r.y, "x": r.x, "alpha": alpha, "beta": beta, "mu": ou.mu,
                        "eq_std": ou.eq_std, "half_life": ou.half_life})
    return out


def select_distance(prices, year, n_pairs=5, formation_months=12):
    """Distance method (Gatev, Goetzmann & Rouwenhorst 2006): over the previous 12 months,
    rank all pairs by the sum of squared differences of their cumulative log returns and
    take the closest `n_pairs`. No significance test. Hedge ratio fixed at 1 (equal dollars);
    the trading threshold is the formation-period std of the difference."""
    start = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(months=formation_months)
    window = prices[(prices.index >= start) & (prices.index < f"{year}-01-01")]
    if len(window) < 200:
        return []
    lw = np.log(window)
    norm = lw - lw.iloc[0]
    cols = list(norm.columns)
    v = norm.values
    scored = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            d = v[:, i] - v[:, j]
            scored.append((float(np.sum(d * d)), cols[i], cols[j]))
    scored.sort()
    out = []
    for _, a, b in scored[:n_pairs]:
        alpha = float(lw[a].iloc[0] - lw[b].iloc[0])      # spread measured relative to the window start
        d = lw[a].values - lw[b].values - alpha
        ou = fit_ou(d)
        out.append({"y": a, "x": b, "alpha": alpha, "beta": 1.0, "mu": float(d.mean()),
                    "eq_std": float(d.std()), "half_life": ou.half_life if ou else np.inf})
    return out


SELECTORS = {"coint": select_cointegration, "distance": select_distance}


def select_all(prices, first_year, last_year, method="coint", **kw):
    """Pair selections for each trading year (computed once, reusable across cost levels)."""
    return {year: SELECTORS[method](prices, year, **kw) for year in range(first_year, last_year + 1)}


def walk_forward(prices: pd.DataFrame, first_year: int, last_year: int,
                 formation_years: int = 3, rules: Rules = Rules(), min_slots: int = 5,
                 fdr: float = 0.05, verbose: bool = True, method: str = "coint",
                 selections: dict | None = None):
    """Run the walk-forward backtest. Returns dict with daily returns, trades, picks, turnover.

    `selections` (from select_all) can be passed in to avoid re-screening for every cost level.
    """
    if selections is None:
        kw = {"formation_years": formation_years, "fdr": fdr} if method == "coint" else {}
        selections = select_all(prices, first_year, last_year, method, **kw)
    logp = np.log(prices)
    daily, all_trades, picks, turnover = [], [], [], []
    for year in range(first_year, last_year + 1):
        trade = logp[(logp.index >= f"{year}-01-01") & (logp.index < f"{year + 1}-01-01")]
        if len(trade) < 20:
            continue
        pairs = selections.get(year, [])
        slots = max(len(pairs), min_slots)
        year_ret = pd.Series(0.0, index=trade.index)
        year_turn = 0.0
        for p in pairs:
            ret, trades, turn = simulate_pair(trade, p["y"], p["x"], p["alpha"], p["beta"],
                                              p["mu"], p["eq_std"], p["half_life"], rules)
            year_ret += ret / slots
            year_turn += turn.sum() / slots
            for tr in trades:
                tr["year"] = year
            all_trades += trades
            picks.append({"year": year, "y": p["y"], "x": p["x"], "beta": p["beta"],
                          "half_life": p["half_life"], "n_trades": len(trades),
                          "pnl": float(np.prod(1 + ret) - 1)})
        daily.append(year_ret)
        turnover.append({"year": year, "turnover": year_turn})
        if verbose:
            print(f"  {year}: {len(pairs)} pairs traded, year return {np.prod(1 + year_ret) - 1:+.2%}")
    returns = pd.concat(daily) if daily else pd.Series(dtype=float)
    return {"returns": returns, "trades": pd.DataFrame(all_trades), "picks": pd.DataFrame(picks),
            "turnover": pd.DataFrame(turnover)}


def performance(returns: pd.Series, trades: pd.DataFrame, turnover: pd.DataFrame) -> dict:
    """Standard summary statistics (252 trading days per year)."""
    if returns.empty:
        return {}
    eq = (1 + returns).cumprod()
    years = len(returns) / 252
    ann_ret = eq.iloc[-1] ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    out = {
        "total_return": eq.iloc[-1] - 1,
        "annual_return": ann_ret,
        "annual_vol": vol,
        "sharpe": returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "days_invested": float((returns != 0).mean()),
        "annual_turnover": float(turnover["turnover"].sum() / years) if len(turnover) else 0.0,
        "n_trades": len(trades),
    }
    if len(trades):
        out["win_rate"] = float((trades["pnl"] > 0).mean())
        out["avg_trade"] = float(trades["pnl"].mean())
        out["avg_days_held"] = float(trades["days"].mean())
        out["exit_reasons"] = trades["exit_reason"].value_counts().to_dict()
    return out
