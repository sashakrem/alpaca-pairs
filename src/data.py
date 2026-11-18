"""Historical daily closes from Alpaca, cached to disk."""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def _client():
    from alpaca.data.historical import StockHistoricalDataClient
    from dotenv import load_dotenv

    load_dotenv()
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. Copy .env.example to .env "
            "and paste your paper-trading keys."
        )
    return StockHistoricalDataClient(key, secret)


def fetch_closes(symbols, start, end=None, feed="sip", use_cache=True) -> pd.DataFrame:
    """Split/dividend-adjusted daily closes, one column per symbol.

    feed="sip" (all US exchanges) works on the free plan for data older than
    15 minutes; use feed="iex" if your account rejects it.
    """
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    symbols = sorted(set(symbols))
    end = pd.Timestamp(end) if end else pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=1))
    start = pd.Timestamp(start)
    key = hashlib.md5(f"{symbols}{start.date()}{end.date()}{feed}".encode()).hexdigest()[:10]
    cache_file = CACHE_DIR / f"closes_{key}.csv"
    if use_cache and cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        adjustment=Adjustment.ALL,
        feed=DataFeed(feed),
    )
    bars = _client().get_stock_bars(req).df
    closes = bars["close"].unstack(level="symbol")
    closes.index = pd.to_datetime(closes.index).tz_convert("America/New_York").tz_localize(None).normalize()

    # Drop symbols with too much missing history (e.g. recent IPOs), then align.
    closes = closes.dropna(axis=1, thresh=int(0.95 * len(closes))).ffill().dropna()
    dropped = set(symbols) - set(closes.columns)
    if dropped:
        print(f"[data] dropped for insufficient history: {sorted(dropped)}")

    CACHE_DIR.mkdir(exist_ok=True)
    closes.to_csv(cache_file)
    return closes
