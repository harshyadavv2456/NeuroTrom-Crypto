"""Synthetic data generator for offline testing when Binance is unreachable."""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA, HISTORY_INTERVALS
from market_data import save_candles

log = logging.getLogger(__name__)


def generate_synthetic_candles(symbol: str, interval: str = "1h", days: int = 90, market_type: str = "spot") -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data."""
    random.seed(hash(symbol) % 2**32)
    np.random.seed(hash(symbol) % 2**32)
    
    end = datetime.now(timezone.utc)
    if interval == "1h":
        periods = days * 24
        freq = "h"
    elif interval == "1d":
        periods = days
        freq = "D"
    elif interval == "1m":
        periods = days * 24 * 60
        freq = "min"
    else:
        periods = days
        freq = "h"
    
    # Generate timestamps
    if interval == "1m":
        # Skip non-trading minutes (crypto trades 24/7, but for realism we keep all)
        ts = pd.date_range(end=end, periods=periods, freq=freq, tz="UTC")
    else:
        ts = pd.date_range(end=end, periods=periods, freq=freq, tz="UTC")
    
    # Price dynamics: geometric brownian motion with jumps
    base_prices = {
        "BTCUSDT": 60000, "ETHUSDT": 3000, "BNBUSDT": 500, "SOLUSDT": 150,
        "XRPUSDT": 0.6, "ADAUSDT": 0.5, "DOGEUSDT": 0.15, "AVAXUSDT": 35,
        "DOTUSDT": 7, "MATICUSDT": 0.7, "LTCUSDT": 80, "LINKUSDT": 15,
        "ATOMUSDT": 8, "UNIUSDT": 6, "XLMUSDT": 0.12, "ALGOUSDT": 0.2,
    }
    base = base_prices.get(symbol, 10.0)
    
    returns = np.random.normal(0.0001, 0.02, len(ts))
    # Add some autocorrelation and volatility clustering
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]
        if abs(returns[i-1]) > 0.03:
            returns[i] *= 1.5
    
    prices = base * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    close = prices
    high = close * (1 + np.random.uniform(0, 0.01, len(ts)))
    low = close * (1 - np.random.uniform(0, 0.01, len(ts)))
    open_p = np.roll(close, 1)
    open_p[0] = close[0] * (1 + np.random.normal(0, 0.005))
    
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_p, close))
    low = np.minimum(low, np.minimum(open_p, close))
    
    volume = np.random.lognormal(10, 1, len(ts))
    # Volume correlates with volatility
    volume = volume * (1 + 3 * np.abs(returns))
    
    df = pd.DataFrame({
        "open_time": ts,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": volume * close,
        "trades": np.random.randint(100, 10000, len(ts)),
        "taker_buy_base": volume * 0.5,
        "taker_buy_quote": volume * close * 0.5,
        "ignore": 0,
    })
    df["symbol"] = symbol
    df["interval"] = interval
    df["market"] = market_type
    df["exchange"] = "synthetic"
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    
    return df


def backfill_synthetic(universe_size: int = 30, days: int = 90):
    """Generate synthetic data for the universe."""
    priority_assets = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
        "DOTUSDT", "MATICUSDT", "AVAXUSDT", "LTCUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT",
        "XLMUSDT", "ALGOUSDT", "VETUSDT", "FILUSDT", "APTUSDT", "ARBUSDT",
        "OPUSDT", "SUIUSDT", "SEIUSDT", "TIUSDT", "NEARUSDT", "AAVEUSDT", "MKRUSDT",
        "RUNEUSDT", "IMXUSDT", "INJUSDT", "PEPEUSDT", "WIFUSDT",
    ]
    symbols = priority_assets[:universe_size]
    
    total = 0
    for sym in symbols:
        for interval in ["1h", "1d"]:
            df = generate_synthetic_candles(sym, interval, days)
            written = save_candles(df, sym, interval, "spot")
            total += written
    
    # Fine-grain for first 15 symbols
    for sym in symbols[:15]:
        df = generate_synthetic_candles(sym, "1m", days=7)
        written = save_candles(df, sym, "1m", "spot")
        total += written
    
    log.info("Synthetic backfill complete: %d rows for %d symbols", total, len(symbols))
    return total


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    backfill_synthetic(60, 90)

