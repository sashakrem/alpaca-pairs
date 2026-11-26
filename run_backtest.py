"""Walk-forward backtest of the pairs strategy (stage 3).

Examples
--------
    python run_backtest.py --universe utilities_wide
    python run_backtest.py --universe all --costs 0 5 10 20
    python run_backtest.py --demo
"""
import argparse
from pathlib import Path

import pandas as pd

from src.backtest import Rules, performance, select_all, walk_forward
from src.universes import UNIVERSES, WIDE_UNIVERSES


def fmt(p):
    if not p:
        return "no trades"
    return (f"Sharpe {p['sharpe']:+.2f} | ann. return {p['annual_return']:+.2%} | vol {p['annual_vol']:.2%} | "
            f"max DD {p['max_drawdown']:.2%} | trades {p['n_trades']} | win {p.get('win_rate', float('nan')):.0%} | "
            f"avg hold {p.get('avg_days_held', float('nan')):.1f}d | turnover {p['annual_turnover']:.1f}x/yr")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--universe", choices=list(UNIVERSES) + ["all"])
    src.add_argument("--demo", action="store_true", help="synthetic market with planted pairs")
    ap.add_argument("--start", default="2016-01-01", help="first date of price history")
    ap.add_argument("--first-year", type=int, default=2019, help="first trading year")
    ap.add_argument("--last-year", type=int, default=2026)
    ap.add_argument("--formation-years", type=int, default=3)
    ap.add_argument("--costs", type=float, nargs="+", default=[5.0], help="cost per side in bps")
    ap.add_argument("--feed", default="sip", choices=["sip", "iex"])
    ap.add_argument("--select", nargs="+", default=["coint"], choices=["coint", "distance"],
                    help="pair selection: coint (statistical screen) and/or distance (GGR benchmark)")
    args = ap.parse_args()

    if args.demo:
        from src.synthetic import synthetic_universe
        prices, _ = synthetic_universe(n_noise=12, n_pairs=3, n_days=2600, seed=1)
        universes = {"demo": prices}
        args.first_year = 2022
    else:
        from src.data import fetch_closes
        names = list(WIDE_UNIVERSES) if args.universe == "all" else [args.universe]
        universes = {n: fetch_closes(UNIVERSES[n], args.start, feed=args.feed) for n in names}

    Path("results").mkdir(exist_ok=True)
    rows = []
    for name, prices in universes.items():
        for method in args.select:
            kw = {"formation_years": args.formation_years} if method == "coint" else {}
            sel = select_all(prices, args.first_year, args.last_year, method, **kw)   # once per method
            for cost in args.costs:
                print(f"\n=== {name}, {method} selection, cost {cost:g} bps/side ===")
                res = walk_forward(prices, args.first_year, args.last_year, rules=Rules(cost_bps=cost),
                                   selections=sel)
                perf = performance(res["returns"], res["trades"], res["turnover"])
                print("  " + fmt(perf))
                rows.append({"universe": name, "select": method, "cost_bps": cost,
                             **{k: v for k, v in perf.items() if k != "exit_reasons"}})
                tag = f"{name}_{method}_{cost:g}bps"
                res["returns"].to_csv(f"results/bt_returns_{tag}.csv")
                res["trades"].to_csv(f"results/bt_trades_{tag}.csv", index=False)
                res["picks"].to_csv(f"results/bt_picks_{tag}.csv", index=False)

    if len(rows) > 1:
        summary = pd.DataFrame(rows)
        summary.to_csv("results/bt_summary.csv", index=False)
        cols = ["universe", "select", "cost_bps", "sharpe", "annual_return", "max_drawdown", "n_trades", "win_rate"]
        with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
            print("\n" + summary[[c for c in cols if c in summary]].to_string(index=False))


if __name__ == "__main__":
    main()
