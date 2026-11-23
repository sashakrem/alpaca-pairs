"""Rebalance an Alpaca PAPER account to target weights.

Safety rules
------------
- Paper only: the client is created with paper=True and there is no switch to turn it off.
- Only symbols the strategy uses are touched. Anything else you hold is left alone
  and its value is excluded from what the strategy manages.
- One rebalance per signal month. Every order is tagged with a client_order_id
  starting "bot-<YYYY-MM>-"; if any such order exists, the month is already done.
  This works from any machine with no local state file.
- Sells go first and must fill before buys are sent, and buys never exceed cash,
  so the account never borrows on margin.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

CASH_BUFFER = 0.01    # keep 1% uninvested for price moves between plan and fill
MIN_TRADE = 25.0      # dollars; skip smaller adjustments
MIN_TRADE_PCT = 0.002 # ...or 0.2% of managed equity, whichever is larger


@dataclass
class Order:
    symbol: str
    side: str            # "buy" | "sell"
    notional: float      # dollars
    full_exit: bool = False


def plan_rebalance(managed_equity: float, holdings: dict[str, float],
                   targets: dict[str, float], universe: list[str]) -> list[Order]:
    """Orders that move `holdings` (symbol -> market value) to `targets` (symbol -> weight).

    Pure function, no API calls, so it is unit-tested. Sells come before buys.
    """
    investable = managed_equity * (1 - CASH_BUFFER)
    min_trade = max(MIN_TRADE, MIN_TRADE_PCT * managed_equity)
    sells, buys = [], []
    for sym in universe:
        have = holdings.get(sym, 0.0)
        want = targets.get(sym, 0.0) * investable
        if want == 0 and have > 0:
            sells.append(Order(sym, "sell", round(have, 2), full_exit=True))
        elif have - want >= min_trade:
            sells.append(Order(sym, "sell", round(have - want, 2)))
        elif want - have >= min_trade:
            buys.append(Order(sym, "buy", round(want - have, 2)))
    return sells + buys


class PaperBroker:
    def __init__(self, key: str, secret: str):
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(key, secret, paper=True)

    # --- account state -------------------------------------------------------
    def account(self):
        return self.client.get_account()

    def positions(self) -> dict:
        return {p.symbol: p for p in self.client.get_all_positions()}

    def market_open(self) -> tuple[bool, datetime]:
        c = self.client.get_clock()
        return c.is_open, c.next_open

    def already_rebalanced(self, period: str) -> list:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        after = datetime.now(timezone.utc) - pd.Timedelta(days=62)
        orders = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL,
                                                         after=after, limit=500))
        return [o for o in orders if (o.client_order_id or "").startswith(f"bot-{period}-")]

    # --- execution -----------------------------------------------------------
    def _submit(self, o: Order, period: str, qty: float | None = None):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        cid = f"bot-{period}-{o.symbol}-{o.side}-{int(time.time() * 1000)}"
        kw = {"qty": qty} if qty is not None else {"notional": o.notional}
        req = MarketOrderRequest(symbol=o.symbol, side=OrderSide(o.side),
                                 time_in_force=TimeInForce.DAY, client_order_id=cid, **kw)
        return self.client.submit_order(req)

    def _wait(self, orders, timeout=120):
        """Poll until every order reaches a final state."""
        done = {"filled", "canceled", "expired", "rejected"}
        deadline = time.time() + timeout
        pending = {o.id for o in orders}
        results = {}
        while pending and time.time() < deadline:
            for oid in list(pending):
                o = self.client.get_order_by_id(oid)
                if str(o.status.value) in done:
                    results[oid] = o
                    pending.discard(oid)
            if pending:
                time.sleep(2)
        if pending:
            raise RuntimeError(f"{len(pending)} order(s) not final after {timeout}s; "
                               "check the Alpaca dashboard before running again.")
        return list(results.values())

    def execute(self, plan: list[Order], period: str, prices: dict[str, float], log=print):
        positions = self.positions()
        sells = [o for o in plan if o.side == "sell"]
        buys = [o for o in plan if o.side == "buy"]

        sent = []
        for o in sells:
            if o.full_exit:   # sell the exact quantity held, fractions included
                sent.append(self._submit(o, period, qty=float(positions[o.symbol].qty)))
            else:
                sent.append(self._submit(o, period))
            log(f"  SELL {o.symbol:5s} ${o.notional:>10,.2f}{'  (exit)' if o.full_exit else ''}")
        for o in self._wait(sent):
            if str(o.status.value) != "filled":
                raise RuntimeError(f"Sell {o.symbol} ended {o.status.value}; not buying.")

        # Spend at most the cash actually available now; never borrow.
        cash = float(self.account().cash)
        want = sum(o.notional for o in buys)
        scale = min(1.0, cash / want) if want > 0 else 1.0
        if scale < 1:
            log(f"  buys scaled to {scale:.1%} to fit available cash ${cash:,.2f}")
        sent = []
        for o in buys:
            o.notional = math.floor(o.notional * scale * 100) / 100
            if o.notional < 1:
                continue
            asset = self.client.get_asset(o.symbol)
            if asset.fractionable:
                sent.append(self._submit(o, period))
            else:  # whole shares only
                qty = math.floor(o.notional / prices[o.symbol])
                if qty < 1:
                    continue
                sent.append(self._submit(o, period, qty=qty))
            log(f"  BUY  {o.symbol:5s} ${o.notional:>10,.2f}")
        filled = self._wait(sent)
        bad = [o for o in filled if str(o.status.value) != "filled"]
        for o in bad:
            log(f"  WARNING: buy {o.symbol} ended {o.status.value}")
