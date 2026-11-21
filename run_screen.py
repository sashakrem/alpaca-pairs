"""Find cointegrated pairs.

Examples
--------
    python run_screen.py --demo                          # synthetic data, no keys needed
    python run_screen.py --universe banks
    python run_screen.py --universe staples --start 2018-01-01 --split 2024-01-01
    python run_screen.py --symbols KO PEP PG CL KMB --fdr 0.10
    python run_screen.py --universe banks --method kalman   # stage 2: time-varying hedge ratio
"""
import argparse
from pathlib import Path

import pandas as pd

from src.screen import screen_pairs, screen_pairs_kalman
from src.universes import UNIVERSES

COLS = {
    "static": ["y", "x", "beta", "p_adj", "half_life", "eq_std",
               "adf_p_holdout", "half_life_holdout", "z_now", "verdict"],
    "kalman": ["y", "x", "beta", "beta_holdout_end", "half_life", "eq_std",
               "adf_p_holdout", "p_adj", "half_life_holdout", "z_now", "verdict"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--universe", choices=UNIVERSES)
    src.add_argument("--symbols", nargs="+")
    src.add_argument("--demo", action="store_true", help="synthetic data with planted pairs")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--split", default="2024-01-01", help="formation/holdout boundary")
    ap.add_argument("--fdr", type=float, default=0.05, help="Benjamini-Hochberg FDR level")
    ap.add_argument("--feed", default="sip", choices=["sip", "iex"])
    ap.add_argument("--method", default="static", choices=["static", "kalman"],
                    help="static = fixed hedge ratio (stage 1); kalman = time-varying (stage 2)")
    ap.add_argument("--beta-drift", type=float, default=0.0,
                    help="--demo only: planted hedge ratios drift by this fraction over the sample")
    args = ap.parse_args()

    if args.demo:
        from src.synthetic import synthetic_universe
        prices, planted = synthetic_universe(beta_drift=args.beta_drift)
        name = "demo"
        print("Planted pairs (ground truth):")
        for p in planted:
            print(f"  {p['pair'][0]}~{p['pair'][1]}  beta={p['beta']:.3f}  half-life={p['half_life']:.1f}d")
    else:
        from src.data import fetch_closes
        symbols = UNIVERSES[args.universe] if args.universe else args.symbols
        name = args.universe or "custom"
        prices = fetch_closes(symbols, args.start, feed=args.feed)

    if args.method == "kalman":
        results, s = screen_pairs_kalman(prices, args.split, fdr=args.fdr)
    else:
        results, s = screen_pairs(prices, args.split, fdr=args.fdr)

    print(f"\nMethod: {args.method}")
    print(f"Formation {s['formation'][0]} -> {s['formation'][1]}   Holdout {s['holdout'][0]} -> {s['holdout'][1]}")
    print(f"{s['symbols']} symbols, {s['pairs_tested']} pairs tested")
    if args.method == "kalman":
        print(f"  half-life in range on formation:          {s['hl_ok_formation']}")
        print(f"  holdout stationary after BH (FDR {args.fdr}):  {s['holds_oos_after_fdr']}")
        print(f"  candidates (also stable on holdout refit): {s['candidates']}\n")
        shown = results[results.verdict != "bad half-life"]   # full table still goes to the CSV
    else:
        shown = results
        print(f"  'significant' at p<0.05 with no correction: {s['naive_significant_at_5pct']}")
        print(f"  significant after BH (FDR {args.fdr}):      {s['significant_after_fdr']}")
        print(f"  candidates (good half-life + holds OOS):  {s['candidates']}\n")

    Path("results").mkdir(exist_ok=True)
    if shown.empty:
        print("No pairs survived. Try another universe, a longer history, or --fdr 0.10.")
        if results.empty:
            return
    else:
        with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
            print(shown[COLS[args.method]].to_string(index=False))

    suffix = "_kalman" if args.method == "kalman" else ""
    out = Path("results") / f"screen_{name}{suffix}.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
