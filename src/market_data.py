"""Market data provider adapters (Binance REST + WebSocket)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable, Awaitable

import numpy as np
import pandas as pd
import requests
import websockets

from config import DATA, HISTORY_INTERVALS, FINE_GRAIN_INTERVALS, FINE_GRAIN_UNIVERSE_SIZE, MAX_BACKFILL_CONCURRENCY, MAX_LIVE_CONCURRENCY
from sources import get_source, approved_sources

log = logging.getLogger(__name__)
RAW = DATA / "raw" / "market"
MARKET = DATA / "market"
STATE = DATA / "state"

BINANCE_REST = "https://api.binance.com"
BINANCE_FUTURES_REST = "https://fapi.binance.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws"
BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"


# ---- REST OHLCV ----

def rest_klines(symbol: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None, limit: int = 1000, market_type: str = "spot") -> pd.DataFrame:
    """Fetch klines from Binance REST API."""
    base = BINANCE_REST if market_type == "spot" else BINANCE_FUTURES_REST
    url = f"{base}/api/v3/klines" if market_type == "spot" else f"{base}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms
    
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    raw = r.json()
    
    if not raw:
        return pd.DataFrame()
    
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["symbol"] = symbol
    df["interval"] = interval
    df["market"] = market_type
    df["exchange"] = "binance"
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    return df


def save_candles(df: pd.DataFrame, symbol: str, interval: str, market_type: str = "spot"):
    """Append candles to Parquet, deduplicating on (symbol, interval, open_time)."""
    if df.empty:
        return 0
    
    raw_dir = RAW / market_type / symbol
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{interval}.parquet"
    
    # Read existing
    existing = pd.DataFrame()
    if raw_path.exists():
        try:
            existing = pd.read_parquet(raw_path)
            if "open_time" in existing.columns:
                existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True)
        except Exception as e:
            log.warning("Could not read existing parquet %s: %s", raw_path, e)
            existing = pd.DataFrame()
    
    combined = pd.concat([existing, df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"], keep="last")
    combined = combined.sort_values("open_time").reset_index(drop=True)
    
    # Validate OHLC consistency
    bad = (combined["high"] < combined["low"]) | (combined["high"] < combined["open"]) | (combined["high"] < combined["close"]) | (combined["low"] > combined["open"]) | (combined["low"] > combined["close"])
    before = len(combined)
    if bad.any():
        log.warning("Dropping %d invalid OHLC rows for %s %s", bad.sum(), symbol, interval)
        combined = combined[~bad]
    written = len(combined) - (len(existing) if not existing.empty else 0)
    if written < 0:
        written = 0
    
    combined.to_parquet(raw_path, index=False, engine="pyarrow")
    
    # Mirror to normalized market dir
    norm_dir = MARKET / market_type / symbol
    norm_dir.mkdir(parents=True, exist_ok=True)
    norm_path = norm_dir / f"{interval}.parquet"
    
    norm_df = combined[["open_time", "open", "high", "low", "close", "volume", "symbol", "interval", "market", "exchange", "ingested_at"]].copy()
    norm_df.columns = ["timestamp", "open", "high", "low", "close", "volume", "symbol", "interval", "market", "exchange", "ingested_at"]
    if norm_path.exists():
        try:
            old = pd.read_parquet(norm_path)
            old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
            norm_df = pd.concat([old, norm_df], ignore_index=True).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
        except Exception:
            pass
    norm_df.to_parquet(norm_path, index=False, engine="pyarrow")
    
    return written


def get_candles(symbol: str, interval: str, market_type: str = "spot", start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
    """Read stored candles from normalized market dir."""
    path = MARKET / market_type / symbol / f"{interval}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    return df.sort_values("timestamp").reset_index(drop=True)


def get_last_timestamp(symbol: str, interval: str, market_type: str = "spot") -> Optional[datetime]:
    df = get_candles(symbol, interval, market_type)
    if df.empty:
        return None
    return df["timestamp"].max().to_pydatetime()


# ---- Backfill ----

def backfill_symbol(symbol: str, interval: str, start: datetime, end: datetime, market_type: str = "spot", checkpoint_every: int = 100) -> int:
    """Resumable backfill for one symbol/interval. Returns rows written."""
    # Load checkpoint
    ckpt_path = STATE / f"backfill_{symbol}_{interval}_{market_type}.json"
    ckpt = {}
    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text())
        except Exception:
            pass
    
    current_start = datetime.fromisoformat(ckpt.get("current_start", start.isoformat()))
    total_rows = ckpt.get("total_rows", 0)
    last_success = ckpt.get("last_success", "")
    
    # Check if already done
    existing_end = get_last_timestamp(symbol, interval, market_type)
    if existing_end and existing_end >= end:
        log.info("Backfill for %s %s %s already complete through %s", symbol, interval, market_type, existing_end)
        return total_rows
    
    cur = max(current_start, start)
    batch = 1000
    consecutive_errors = 0
    
    while cur < end:
        if batch > 1000:
            batch = 1000
        start_ms = int(cur.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        
        try:
            df = rest_klines(symbol, interval, start_ms=start_ms, end_ms=end_ms, limit=batch, market_type=market_type)
            if df.empty:
                break
            
            written = save_candles(df, symbol, interval, market_type)
            total_rows += written
            new_end = df["open_time"].max().to_pydatetime()
            cur = new_end + pd.Timedelta(milliseconds=1)
            consecutive_errors = 0
            
            # Save checkpoint
            ckpt_path.write_text(json.dumps({
                "current_start": cur.isoformat(),
                "total_rows": total_rows,
                "last_success": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "interval": interval,
                "market": market_type,
            }))
            
            if total_rows % (checkpoint_every * batch) == 0 and total_rows > 0:
                log.info("Backfill %s %s: %d rows, up to %s", symbol, interval, total_rows, cur)
            
            # Respect rate limits
            time.sleep(0.05)
            
        except requests.exceptions.HTTPError as e:
            consecutive_errors += 1
            if e.response is not None and e.response.status_code == 429:
                wait = min(60, 2 ** consecutive_errors)
                log.warning("Rate limited, backing off %ds", wait)
                time.sleep(wait)
            elif e.response is not None and e.response.status_code >= 500:
                wait = min(30, 2 ** consecutive_errors)
                log.warning("Server error %d, retrying in %ds", e.response.status_code, wait)
                time.sleep(wait)
            else:
                log.error("HTTP error backfilling %s %s: %s", symbol, interval, e)
                break
        except Exception as e:
            consecutive_errors += 1
            log.error("Error backfilling %s %s: %s", symbol, interval, e)
            time.sleep(1)
            if consecutive_errors > 5:
                break
    
    log.info("Backfill complete for %s %s: %d total rows", symbol, interval, total_rows)
    return total_rows


# ---- WebSocket Live ----

async def _ws_listen(streams: list[str], callback: Callable[[dict], Awaitable[None]], market_type: str = "spot", max_idle: int = 300):
    """Listen to Binance WebSocket streams."""
    url = f"{BINANCE_WS}/{''.join(s if s.startswith('/') else '/' + s for s in streams)}" if market_type == "spot" else f"{BINANCE_FUTURES_WS}/{''.join(s if s.startswith('/') else '/' + s for s in streams)}"
    
    backoff = 1
    while True:
        try:
            log.info("Connecting WS: %s", url[:80])
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=10) as ws:
                backoff = 1
                last_msg = time.time()
                async for msg in ws:
                    last_msg = time.time()
                    data = json.loads(msg)
                    if "e" in data and data["e"] == "kline":
                        k = data["k"]
                        row = {
                            "symbol": k["s"],
                            "interval": k["i"],
                            "open_time": pd.to_datetime(k["t"], unit="ms", utc=True).isoformat(),
                            "open": float(k["o"]),
                            "high": float(k["h"]),
                            "low": float(k["l"]),
                            "close": float(k["c"]),
                            "volume": float(k["v"]),
                            "is_closed": k["x"],
                            "exchange": "binance",
                            "market": market_type,
                            "ingested_at": datetime.now(timezone.utc).isoformat(),
                        }
                        await callback(row)
                    
                    if time.time() - last_msg > max_idle:
                        log.warning("WS idle for %ds, reconnecting", max_idle)
                        break
        except Exception as e:
            log.warning("WS error: %s, reconnecting in %ds", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


# ---- Public API ----

def fetch_ticker(symbol: str, market_type: str = "spot") -> dict:
    """Fetch 24h ticker stats."""
    base = BINANCE_REST if market_type == "spot" else BINANCE_FUTURES_REST
    url = f"{base}/api/v3/ticker/24hr" if market_type == "spot" else f"{base}/fapi/v1/ticker/24hr"
    r = requests.get(url, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    d = r.json()
    return {
        "symbol": d["symbol"],
        "price": float(d["lastPrice"]),
        "change_pct": float(d["priceChangePercent"]),
        "volume": float(d["volume"]),
        "high": float(d["highPrice"]),
        "low": float(d["lowPrice"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    # Quick test
    df = rest_klines("BTCUSDT", "1h", limit=5)
    print(df[["open_time", "open", "close", "volume"]].to_string(index=False))

