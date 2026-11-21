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

import os
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


# ---------------------------------------------------------------------------
# Stage 2: Kalman screen (time-varying hedge ratio, OU spread as a state)
# ---------------------------------------------------------------------------

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


def portfolio_spread(y, x, beta_prev):
    """Value of the self-financing hedged portfolio: long 1 unit of y, short beta of x,
    rebalanced daily using the hedge ratio known the day before.

        V_t = sum_{s<=t} (dy_s - beta_{s-1} dx_s)

    This is the spread you can actually trade. Testing y_t - beta_t x_t instead is a
    trap: with log prices around 4, small moves in beta_t act like a free intercept
    and make unrelated stocks look mean-reverting. V_t can't profit from beta moving.
    """
    dy, dx = np.diff(y), np.diff(x)
    return np.concatenate([[0.0], np.cumsum(dy - beta_prev[:-1] * dx)])


def _kalman_pair(args):
    """Fit one pair in both directions on the formation period, keep the direction
    whose portfolio spread has the shorter formation half-life (formation data only,
    so the holdout test stays clean), then run the frozen model through the holdout."""
    from .kalman import fit_kalman, kalman_filter

    a, b, form_a, form_b, hold_a, hold_b = args
    fits = []
    for y, x, fy, fx, hy, hx in ((a, b, form_a, form_b, hold_a, hold_b),
                                 (b, a, form_b, form_a, hold_b, hold_a)):
        p, kf = fit_kalman(fy, fx)
        ou = fit_ou(portfolio_spread(fy, fx, kf.beta)[20:])
        fits.append((ou.half_life if ou else np.inf, y, x, p, kf, ou, hy, hx))
    hl, y, x, p, kf, ou, hy, hx = min(fits, key=lambda f: f[0])
    row = {"y": y, "x": x, "beta": kf.beta[-1], "half_life": hl,
           "half_life_mle": p.half_life, "q": p.q,
           "eq_std": ou.eq_std if ou else np.nan,
           "adf_p_holdout": 1.0, "half_life_holdout": np.nan,
           "beta_holdout_end": np.nan, "z_now": np.nan}
    if len(hy) >= 30:
        # Parameters frozen; the filter keeps updating beta day by day, exactly as it
        # would when trading live (each day's P&L uses the previous day's beta).
        kh = kalman_filter(hy, hx, p, s0=kf.state, P0=kf.P, burn_in=0)
        v = portfolio_spread(hy, hx, kh.beta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # "ct": allow a slow linear drift (e.g. a fee gap between two ETFs)
            row["adf_p_holdout"] = adfuller(v, regression="ct", autolag="AIC")[1]
        ou_h = fit_ou(v)
        row["half_life_holdout"] = ou_h.half_life if ou_h else np.inf
        row["beta_holdout_end"] = kh.beta[-1]
        if ou_h:
            row["z_now"] = float(ou_h.zscore(v[-1]))
    return row


def screen_pairs_kalman(
    prices: pd.DataFrame,
    split_date,
    fdr: float = 0.05,
    min_half_life: float = 1.0,
    max_half_life: float = 60.0,
    workers: int | None = None,
):
    """Kalman screen. A pair is a CANDIDATE if
       1. formation half-life is in [min, max]                (it mean-reverts, at a tradeable speed)
       2. the holdout spread from the frozen model passes ADF,
          Benjamini-Hochberg-corrected across ALL pairs        (it still mean-reverts on unseen data)
       3. the holdout portfolio spread's own OU half-life is in range (the dynamics didn't change)
    All three are measured on the tradeable portfolio spread (see portfolio_spread).
    Returns (results DataFrame, summary dict)."""
    from concurrent.futures import ProcessPoolExecutor

    split_date = pd.Timestamp(split_date)
    logp = np.log(prices.dropna(axis=1))
    form = logp[logp.index < split_date]
    hold = logp[logp.index >= split_date]
    if len(form) < 250:
        raise ValueError(f"Formation period has only {len(form)} rows; need >= 250.")

    jobs = [(a, b, form[a].values, form[b].values, hold[a].values, hold[b].values)
            for a, b in combinations(logp.columns, 2)]
    # One math-library thread per worker. Otherwise every process starts its own
    # BLAS thread pool and they fight over the cores (load average in the hundreds).
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: "1" for k in _THREAD_VARS})
    try:
        with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as ex:
            out = pd.DataFrame(list(ex.map(_kalman_pair, jobs, chunksize=4)))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    in_range = lambda h: (h >= min_half_life) & (h <= max_half_life)
    out["hl_ok"] = in_range(out["half_life"])
    p = np.where(out["hl_ok"], out["adf_p_holdout"], 1.0)   # failed step 1 -> can't pass step 2
    out["holds_oos"], out["p_adj"] = benjamini_hochberg(p, fdr)
    out["stable"] = in_range(out["half_life_holdout"])
    out["verdict"] = np.select(
        [~out["hl_ok"], ~out["holds_oos"], ~out["stable"]],
        ["bad half-life", "broke out-of-sample", "dynamics changed"], "CANDIDATE")
    order = {"CANDIDATE": 0, "dynamics changed": 1, "broke out-of-sample": 2, "bad half-life": 3}
    out = out.sort_values(["verdict", "p_adj"], key=lambda c: c.map(order) if c.name == "verdict" else c)
    out = out.reset_index(drop=True)

    summary = {
        "symbols": logp.shape[1],
        "pairs_tested": len(out),
        "hl_ok_formation": int(out["hl_ok"].sum()),
        "holds_oos_after_fdr": int(out["holds_oos"].sum()),
        "candidates": int((out["verdict"] == "CANDIDATE").sum()),
        "formation": (form.index[0].date(), form.index[-1].date()),
        "holdout": (hold.index[0].date(), hold.index[-1].date()) if len(hold) else None,
    }
    return out, summary
