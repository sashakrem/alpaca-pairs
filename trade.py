"""Rebalance your Alpaca PAPER account to the strategy's current target.

    python trade.py                     # dry run: show account, target and planned orders
    python trade.py --execute           # actually place the orders (paper money)
    python trade.py --strategy spy      # use another strategy

Run it once each weekday while the market is open (e.g. 10:00 New York time). It
trades only on the first run after each month end; on other days it reports and exits.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from bot.broker import PaperBroker, plan_rebalance
from bot.data import alpaca_closes, alpaca_keys
from bot.strategy import STRATEGIES, target_weights


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="momentum", choices=STRATEGIES)
    ap.add_argument("--execute", action="store_true", help="place orders (default: dry run)")
    ap.add_argument("--force", action="store_true", help="rebalance even if this month is already done")
    args = ap.parse_args()

    today = datetime.now(ZoneInfo("America/New_York"))
    _, universe = STRATEGIES[args.strategy]
    closes = alpaca_closes(universe)
    signal_date, targets = target_weights(args.strategy, closes, today=today.date())
    period = f"{signal_date:%Y-%m}"

    broker = PaperBroker(*alpaca_keys())
    acct = broker.account()
    positions = broker.positions()
    holdings = {s: float(p.market_value) for s, p in positions.items() if s in universe}
    others = {s: float(p.market_value) for s, p in positions.items() if s not in universe}
    managed = float(acct.equity) - sum(others.values())

    print(f"{today:%Y-%m-%d %H:%M} ET   strategy: {args.strategy}   signal from {signal_date:%Y-%m-%d} close")
    print(f"Paper account equity ${float(acct.equity):,.2f}   cash ${float(acct.cash):,.2f}")
    if others:
        print(f"Not managed (left alone): {', '.join(f'{s} ${v:,.0f}' for s, v in others.items())}")
    print("\nsymbol   target     now")
    for s in sorted(set(targets.index) | set(holdings)):
        print(f"{s:6s} {targets.get(s, 0):7.1%} {holdings.get(s, 0) / managed:7.1%}")

    plan = plan_rebalance(managed, holdings, targets.to_dict(), universe)
    if not plan:
        print("\nAlready on target. Nothing to do.")
        return
    print("\nPlanned orders:")
    for o in plan:
        print(f"  {o.side.upper():4s} {o.symbol:5s} ${o.notional:>10,.2f}{'  (exit)' if o.full_exit else ''}")

    done = broker.already_rebalanced(period)
    if done and not args.force:
        print(f"\nAlready rebalanced for the {period} signal ({len(done)} orders). "
              "Waiting for next month end. (--force overrides.)")
        return
    if not args.execute:
        print("\nDry run. Add --execute to place these orders.")
        return
    is_open, next_open = broker.market_open()
    if not is_open:
        print(f"\nMarket is closed (next open {next_open}). Run again during market hours.")
        return
    print("\nPlacing orders:")
    broker.execute(plan, period, prices=closes.iloc[-1].to_dict())
    print("Done.")


if __name__ == "__main__":
    main()
