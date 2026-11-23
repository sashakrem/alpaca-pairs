"""Daily prices (split- and dividend-adjusted).

Backtests use Yahoo Finance: free, no key, long history.
The live bot uses Alpaca's own market data with your keys.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data"


def yahoo_prices(symbols, start="2006-01-01", use_cache=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(opens, closes), one column per symbol, cached per day."""
    import yfinance as yf

    symbols = sorted(set(symbols))
    f = CACHE / f"yahoo_{'_'.join(symbols)}_{start}_{datetime.now():%Y%m%d}.pkl"
    if use_cache and f.exists():
        return pd.read_pickle(f)
    df = yf.download(symbols, start=start, auto_adjust=True, progress=False)
    opens, closes = df["Open"][symbols].dropna(), df["Close"][symbols].dropna()
    idx = opens.index.intersection(closes.index)
    out = opens.loc[idx], closes.loc[idx]
    CACHE.mkdir(exist_ok=True)
    pd.to_pickle(out, f)
    return out


def alpaca_keys() -> tuple[str, str]:
    from dotenv import load_dotenv

    load_dotenv()
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. "
                         "Copy .env.example to .env and paste your PAPER keys.")
    return key, secret


def alpaca_closes(symbols, days: int = 500) -> pd.DataFrame:
    """Adjusted daily closes from Alpaca (SIP feed; free for data older than 15 minutes)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    req = StockBarsRequest(symbol_or_symbols=sorted(set(symbols)), timeframe=TimeFrame.Day,
                           start=end - timedelta(days=days), end=end,
                           adjustment=Adjustment.ALL, feed=DataFeed.SIP)
    bars = StockHistoricalDataClient(*alpaca_keys()).get_stock_bars(req).df
    closes = bars["close"].unstack(level="symbol")
    closes.index = (pd.to_datetime(closes.index).tz_convert("America/New_York")
                    .tz_localize(None).normalize())
    return closes.ffill().dropna()
