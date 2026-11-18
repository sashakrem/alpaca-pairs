"""Cointegration screening with multiple-testing control and out-of-sample checks.

Pipeline
--------
1. Split prices into a formation period (to find pairs) and a holdout period
   (to check whether the relationship survived).
2. For every pair, run Engle-Granger in both directions. Take the better
   direction and Bonferroni-correct it (x2) so picking a direction isn't free.
3. Control the false discovery rate across all pairs with Benjamini-Hochberg.
   Testing 190 pairs at alpha=0.05 gives ~10 false "cointegrated" pairs by
   chance alone, so this step matters.
4. For surviving pairs, fit the hedge ratio and an OU model to the spread.
5. Freeze those parameters and test the holdout spread for stationarity (ADF).
"""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from .ou import fit_ou


def hedge_regression(y, x):
    """OLS of y on [1, x]. Returns (alpha, beta)."""
    X = np.column_stack([np.ones_like(x), x])
    (alpha, beta), *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(alpha), float(beta)


def benjamini_hochberg(pvals, q=0.05):
    """Return (reject mask, BH-adjusted p-values)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], bool), np.array([])
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate((ranked * m / ranks)[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj <= q, adj


def engle_granger_best(logp: pd.DataFrame, a: str, b: str):
    """Run EG both ways; return (y, x, direction-corrected p-value)."""
    ya, xb = logp[a].values, logp[b].values
    p_ab = coint(ya, xb)[1]
    p_ba = coint(xb, ya)[1]
    if p_ab <= p_ba:
        return a, b, min(1.0, 2 * p_ab)
    return b, a, min(1.0, 2 * p_ba)


def screen_pairs(
    prices: pd.DataFrame,
    split_date,
    fdr: float = 0.05,
    min_half_life: float = 2.0,
    max_half_life: float = 60.0,
    holdout_alpha: float = 0.05,
):
    """Screen all pairs in `prices` (columns = symbols, index = dates).

    Returns (results DataFrame, summary dict).
    """
    split_date = pd.Timestamp(split_date)
    logp = np.log(prices.dropna(axis=1))
    form = logp[logp.index < split_date]
    hold = logp[logp.index >= split_date]
    if len(form) < 250:
        raise ValueError(f"Formation period has only {len(form)} rows; need >= 250.")

    tests = []
    for a, b in combinations(logp.columns, 2):
        y, x, p = engle_granger_best(form, a, b)
        tests.append({"y": y, "x": x, "p_value": p})
    tests = pd.DataFrame(tests)
    tests["significant"], tests["p_adj"] = benjamini_hochberg(tests["p_value"], fdr)

    rows = []
    for r in tests[tests["significant"]].itertuples():
        alpha, beta = hedge_regression(form[r.y].values, form[r.x].values)
        spread_f = form[r.y].values - beta * form[r.x].values - alpha
        ou = fit_ou(spread_f)
        if ou is None:
            continue
        row = {
            "y": r.y, "x": r.x, "beta": beta, "alpha": alpha,
            "p_adj": r.p_adj, "half_life": ou.half_life,
            "theta": ou.theta, "mu": ou.mu, "sigma": ou.sigma, "eq_std": ou.eq_std,
            "adf_p_holdout": np.nan, "half_life_holdout": np.nan, "z_now": np.nan,
        }
        if len(hold) >= 30:
            # Frozen formation-period coefficients -> no look-ahead.
            spread_h = hold[r.y].values - beta * hold[r.x].values - alpha
            with warnings.catch_warnings():  # statsmodels return-type FutureWarning
                warnings.simplefilter("ignore", FutureWarning)
                row["adf_p_holdout"] = adfuller(spread_h, autolag="AIC")[1]
            ou_h = fit_ou(spread_h)
            row["half_life_holdout"] = ou_h.half_life if ou_h else np.inf
            row["z_now"] = float(ou.zscore(spread_h[-1]))
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["hl_ok"] = out["half_life"].between(min_half_life, max_half_life)
        out["holds_oos"] = out["adf_p_holdout"] < holdout_alpha
        out["verdict"] = np.where(
            out["hl_ok"] & out["holds_oos"], "CANDIDATE",
            np.where(~out["hl_ok"], "bad half-life", "broke out-of-sample"),
        )
        out = out.sort_values(["verdict", "p_adj"]).reset_index(drop=True)

    summary = {
        "symbols": logp.shape[1],
        "pairs_tested": len(tests),
        "significant_after_fdr": int(tests["significant"].sum()),
        "naive_significant_at_5pct": int((tests["p_value"] < 0.05).sum()),
        "candidates": int((out["verdict"] == "CANDIDATE").sum()) if not out.empty else 0,
        "formation": (form.index[0].date(), form.index[-1].date()),
        "holdout": (hold.index[0].date(), hold.index[-1].date()) if len(hold) else None,
    }
    return out, summary
