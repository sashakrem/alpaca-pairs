"""Regenerate the README figures in docs/figures/ from backtest results and cached data.

    python run_backtest.py --universe all --select distance --costs 0 5 20
    python make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import fetch_closes
from src.ou import fit_ou
from src.screen import hedge_regression
from src.universes import UNIVERSES, WIDE_UNIVERSES

OUT = Path("docs/figures")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def title(ax, main, sub):
    ax.set_title(main, loc="left", fontsize=14, fontweight="bold", color=INK, pad=26)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=10.5, va="bottom")


def equity_curves():
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, cost in enumerate((0, 5, 20)):
        rs = pd.concat([pd.read_csv(f"results/bt_returns_{u}_distance_{cost}bps.csv",
                                    index_col=0, parse_dates=True).iloc[:, 0] for u in WIDE_UNIVERSES],
                       axis=1).fillna(0).mean(axis=1)
        eq = (1 + rs).cumprod() - 1
        sharpe = rs.mean() / rs.std() * np.sqrt(252)
        ax.plot(eq.index, eq * 100, color=SERIES[i], lw=2, label=f"{cost} bp/side")
        ax.annotate(f"{cost} bp  (Sharpe {sharpe:+.2f})", (eq.index[-1], eq.iloc[-1] * 100),
                    xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylabel("cumulative return, %")
    ax.legend(loc="lower left", frameon=False, labelcolor=INK2)
    ax.set_xlim(pd.Timestamp("2018-11-01"), pd.Timestamp("2028-06-01"))
    ax.set_xticks([pd.Timestamp(f"{yr}-01-01") for yr in range(2019, 2027)], [str(yr) for yr in range(2019, 2027)])
    title(ax, "Distance-method pairs trading, 6 universes combined",
          "Walk-forward 2019–2026. Roughly flat before costs; realistic costs turn it negative.")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curves.png", dpi=150)


def funnel():
    stages = [("pair-years tested", 8672), ("p < 0.05, uncorrected", 562),
              ("after Benjamini-Hochberg", 21), ("also held up in validation year", 1)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(stages))[::-1]
    ax.barh(y, [v for _, v in stages], color=BLUE_RAMP, height=0.62, edgecolor=SURFACE, linewidth=2)
    for yi, (lab, v) in zip(y, stages):
        ax.text(v * 1.15, yi, f"{v:,}", va="center", color=INK, fontsize=11, fontweight="bold")
    ax.set_yticks(y, [s for s, _ in stages])
    ax.set_xscale("log")
    ax.set_xlim(0.8, 40000)
    ax.axvline(8672 * 0.05, color=MUTED, lw=1.2, ls="--")
    ax.text(8672 * 0.05 * 1.12, y[2], "← expected by chance\n    alone (5%): 434",
            color=INK2, fontsize=9.5, va="center")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("count (log scale)")
    title(ax, "The statistical screen finds almost nothing",
          "6 universes × 8 yearly screens. Significance barely exceeds what pure noise produces.")
    fig.tight_layout()
    fig.savefig(OUT / "screen_funnel.png", dpi=150)


def broken_pair(y="CL", x="KHC"):
    p = np.log(fetch_closes(UNIVERSES["staples"], "2019-01-01"))
    form = p[p.index < "2024-01-01"]
    a, b = hedge_regression(form[y].values, form[x].values)
    ou = fit_ou(form[y].values - b * form[x].values - a)
    z = pd.Series(ou.zscore((p[y] - b * p[x] - a).values), index=p.index)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axvspan(p.index[0], pd.Timestamp("2024-01-01"), color="#f0efec", lw=0)
    ax.text(p.index[20], 7.6, "fit period: parameters estimated here", color=INK2, fontsize=9.5)
    ax.text(pd.Timestamp("2024-02-01"), 7.6, "unseen data", color=INK2, fontsize=9.5)
    for lvl in (-2, 2):
        ax.axhline(lvl, color=MUTED, lw=1, ls="--")
    ax.text(p.index[-1], 2.25, "entry ±2", color=INK2, fontsize=9.5, ha="right")
    ax.plot(z.index, z, color=SERIES[0], lw=1.6)
    ax.set_ylim(-4.6, 8.3)
    ax.set_ylabel("spread z-score")
    title(ax, f"A pair that broke: {y} vs {x}",
          f"Cointegrated 2019–2023 (half-life {ou.half_life:.0f} days), then drifted to z ≈ {z.iloc[-1]:.0f}. "
          "A naive bot keeps selling into it.")
    fig.tight_layout()
    fig.savefig(OUT / "broken_pair.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    equity_curves()
    funnel()
    broken_pair()
    print(f"saved figures to {OUT}/")
