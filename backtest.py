"""Backtest the strategies on historical data and compare them with buying SPY.

    python backtest.py                         # all strategies, 2008 -> today
    python backtest.py --start 2017-01-01      # a later window
    python backtest.py --cost-bps 20           # harsher trading-cost assumption
"""
import argparse
from pathlib import Path

import pandas as pd

from bot.backtest import run_backtest, summarize
from bot.data import yahoo_prices
from bot.strategy import STRATEGIES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    args = ap.parse_args()

    symbols = sorted({s for _, syms in STRATEGIES.values() for s in syms})
    opens, closes = yahoo_prices(symbols)

    # Start every strategy on the same day: the first day ALL of them can trade
    # (momentum needs 12 months of history for every ETF first).
    start = max(run_backtest(opens, closes, n, start=args.start).index[0] for n in STRATEGIES)

    stats, curves = {}, {}
    for name in STRATEGIES:
        bt = run_backtest(opens, closes, name, start=start,
                          capital=args.capital, cost_bps=args.cost_bps)
        curves[name] = bt["equity"]
        stats[name] = summarize(bt["equity"]) | {"costs": bt["costs"].sum()}

    s0 = next(iter(stats.values()))
    print(f"${args.capital:,.0f} from {s0['start']} to {s0['end']}, trading cost {args.cost_bps:g} bps\n")
    print(f"{'strategy':10s} {'final $':>12s} {'yearly':>7s} {'vol':>6s} {'Sharpe':>6s} "
          f"{'max drop':>8s} {'worst yr':>8s} {'losing yrs':>10s} {'costs $':>9s}")
    for name, s in stats.items():
        print(f"{name:10s} {s['final']:>12,.0f} {s['cagr']:>7.1%} {s['vol']:>6.1%} {s['sharpe']:>6.2f} "
              f"{s['max_drawdown']:>8.1%} {s['worst_year']:>8.1%} "
              f"{s['losing_years']:>4d} of {s['years']:<3d} {s['costs']:>9,.0f}")

    yearly = pd.DataFrame({n: s["yearly"] for n, s in stats.items()})
    yearly.index = yearly.index.year
    print("\nReturn by calendar year:")
    print(yearly.map(lambda v: f"{v:6.1%}").to_string())

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(curves).to_csv("results/equity_curves.csv")
    print("\nSaved results/equity_curves.csv")


if __name__ == "__main__":
    main()
