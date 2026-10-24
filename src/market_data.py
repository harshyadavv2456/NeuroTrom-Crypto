"""Market data provider adapters (Binance REST + WebSocket, Kraken REST)."""
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

from config import DATA, HISTORY_INTERVALS, FINE_GRAIN_INTERVALS, FINE_GRAIN_UNIVERSE_SIZE, MAX_BACKFILL_CONCURRENCY, MAX_LIVE_CONCURRENCY, SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC
from sources import get_source, approved_sources

log = logging.getLogger(__name__)
RAW = DATA / "raw" / "market"
MARKET = DATA / "market"
SYNTHETIC = DATA / "synthetic"
STATE = DATA / "state"

BINANCE_REST = "https://api.binance.com"
BINANCE_FUTURES_REST = "https://fapi.binance.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws"
BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"
KRAKEN_REST = "https://api.kraken.com/0/public"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _add_provenance(df: pd.DataFrame, source_mode: str, provider: str) -> pd.DataFrame:
    """Add provenance columns to a DataFrame of candles."""
    if "ingested_at" not in df.columns:
        df["ingested_at"] = _now_iso()
    if "source_mode" not in df.columns:
        df["source_mode"] = source_mode
    if "provider" not in df.columns:
        df["provider"] = provider
    return df


# ---- Binance REST OHLCV ----

def rest_klines(symbol: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None, limit: int = 1000, market_type: str = "spot") -> pd.DataFrame:
    """Fetch klines from Binance REST API. Returns empty DataFrame on failure."""
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
    df = _add_provenance(df, SOURCE_HISTORICAL, "binance")
    return df


def save_candles(df: pd.DataFrame, symbol: str, interval: str, market_type: str = "spot", source_mode: str = SOURCE_HISTORICAL, provider: str = "binance"):
    """Append candles to Parquet, deduplicating on (symbol, interval, open_time)."""
    if df.empty:
        return 0
    
    raw_dir = RAW / market_type / symbol
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{interval}.parquet"
    
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
    
    bad = (combined["high"] < combined["low"]) | (combined["high"] < combined["open"]) | (combined["high"] < combined["close"]) | (combined["low"] > combined["open"]) | (combined["low"] > combined["close"])
    before = len(combined)
    if bad.any():
        log.warning("Dropping %d invalid OHLC rows for %s %s", bad.sum(), symbol, interval)
        combined = combined[~bad]
    written = len(combined) - (len(existing) if not existing.empty else 0)
    if written < 0:
        written = 0
    
    combined.to_parquet(raw_path, index=False, engine="pyarrow")
    
    norm_dir = MARKET / market_type / symbol
    norm_dir.mkdir(parents=True, exist_ok=True)
    norm_path = norm_dir / f"{interval}.parquet"
    
    norm_df = combined[["open_time", "open", "high", "low", "close", "volume", "symbol", "interval", "market", "exchange", "ingested_at", "source_mode", "provider"]].copy()
    norm_df.columns = ["timestamp", "open", "high", "low", "close", "volume", "symbol", "interval", "market", "exchange", "ingested_at", "source_mode", "provider"]
    if norm_path.exists():
        try:
            old = pd.read_parquet(norm_path)
            old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
            norm_df = pd.concat([old, norm_df], ignore_index=True).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
        except Exception:
            pass
    norm_df.to_parquet(norm_path, index=False, engine="pyarrow")
    
    return written


def get_candles(symbol: str, interval: str, market_type: str = "spot", start: Optional[datetime] = None, end: Optional[datetime] = None, source_mode: Optional[str] = None) -> pd.DataFrame:
    """Read stored candles from normalized market dir. Optionally filter by source_mode."""
    path = MARKET / market_type / symbol / f"{interval}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if source_mode is not None and "source_mode" in df.columns:
        df = df[df["source_mode"] == source_mode]
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    return df.sort_values("timestamp").reset_index(drop=True)


def get_last_timestamp(symbol: str, interval: str, market_type: str = "spot", source_mode: Optional[str] = None) -> Optional[datetime]:
    df = get_candles(symbol, interval, market_type, source_mode=source_mode)
    if df.empty:
        return None
    return df["timestamp"].max().to_pydatetime()


# ---- Kraken REST OHLCV ----

def kraken_ohlc(pair: str, interval: int = 60, since: Optional[int] = None) -> pd.DataFrame:
    """Fetch OHLC from Kraken public API."""
    url = f"{KRAKEN_REST}/OHLC"
    params = {"pair": pair, "interval": interval}
    if since is not None:
        params["since"] = since
    
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    if data.get("error"):
        raise ValueError(f"Kraken error: {data['error']}")
    
    result = data.get("result", {})
    # Kraken returns data with the pair name as key, and a 'last' field
    pair_key = [k for k in result.keys() if k != "last"]
    if not pair_key:
        return pd.DataFrame()
    
    raw = result[pair_key[0]]
    cols = ["time", "open", "high", "low", "close", "vwap", "volume", "count"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "vwap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["symbol"] = pair
    df["interval"] = f"{interval // 60}h" if interval >= 60 else f"{interval}m"
    df["market"] = "spot"
    df["exchange"] = "kraken"
    df = _add_provenance(df, SOURCE_HISTORICAL, "kraken")
    return df


# ---- Backfill ----

def backfill_symbol(symbol: str, interval: str, start: datetime, end: datetime, market_type: str = "spot", checkpoint_every: int = 100, provider: str = "binance") -> int:
    """Resumable backfill for one symbol/interval. Returns rows written."""
    ckpt_path = STATE / f"backfill_{symbol}_{interval}_{market_type}_{provider}.json"
    ckpt = {}
    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text())
        except Exception:
            pass
    
    current_start = datetime.fromisoformat(ckpt.get("current_start", start.isoformat()))
    total_rows = ckpt.get("total_rows", 0)
    
    existing_end = get_last_timestamp(symbol, interval, market_type)
    if existing_end and existing_end >= end - pd.Timedelta(hours=2):
        log.info("Backfill for %s %s %s already complete through %s", symbol, interval, market_type, existing_end)
        return total_rows
    
    cur = max(current_start, start)
    batch = 1000
    consecutive_errors = 0
    
    while cur < end:
        start_ms = int(cur.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        
        try:
            if provider == "binance":
                df = rest_klines(symbol, interval, start_ms=start_ms, end_ms=end_ms, limit=batch, market_type=market_type)
            elif provider == "kraken":
                # Kraken interval is in minutes
                kraken_interval = 60 if interval == "1h" else 1440 if interval == "1d" else 60
                df = kraken_ohlc(symbol, interval=kraken_interval, since=start_ms // 1000)
                if not df.empty:
                    # Filter to requested range
                    df = df[(df["time"] >= pd.Timestamp(cur, tz="UTC")) & (df["time"] <= pd.Timestamp(end, tz="UTC"))]
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            if df.empty:
                break
            
            written = save_candles(df, symbol, interval, market_type, source_mode=SOURCE_HISTORICAL, provider=provider)
            total_rows += written
            new_end = df["open_time"].max().to_pydatetime() if "open_time" in df.columns else df["time"].max().to_pydatetime()
            cur = new_end + pd.Timedelta(milliseconds=1)
            consecutive_errors = 0
            
            ckpt_path.write_text(json.dumps({
                "current_start": cur.isoformat(),
                "total_rows": total_rows,
                "last_success": _now_iso(),
                "symbol": symbol,
                "interval": interval,
                "market": market_type,
                "provider": provider,
            }))
            
            time.sleep(0.05)
            
        except requests.exceptions.HTTPError as e:
            consecutive_errors += 1
            if e.response is not None and e.response.status_code == 429:
                wait = min(60, 2 ** consecutive_errors)
                log.warning("Rate limited, backing off %ds", wait)
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
    
    log.info("Backfill complete for %s %s via %s: %d total rows", symbol, interval, provider, total_rows)
    return total_rows


# ---- WebSocket Live ----

async def _ws_listen(streams: list[str], callback: Callable[[dict], Awaitable[None]], market_type: str = "spot", max_idle: int = 300):
    url = f"{BINANCE_WS}/{''.join(s if s.startswith('/') else '/' + s for s in streams)}" if market_type == "spot" else f"{BINANCE_FUTURES_WS}/{''.join(s if s.startswith('/') else '/' + s for s in streams)}"
    backoff = 1
    while True:
        try:
            log.info("Connecting WS: %s", url[:80])
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=3) as ws:
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
                            "ingested_at": _now_iso(),
                            "source_mode": SOURCE_LIVE,
                            "provider": "binance",
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

def fetch_ticker(symbol: str, market_type: str = "spot", provider: str = "binance") -> dict:
    """Fetch 24h ticker stats."""
    if provider == "binance":
        base = BINANCE_REST if market_type == "spot" else BINANCE_FUTURES_REST
        url = f"{base}/api/v3/ticker/24hr" if market_type == "spot" else f"{base}/fapi/v1/ticker/24hr"
        r = requests.get(url, params={"symbol": symbol}, timeout=3)
        r.raise_for_status()
        d = r.json()
        return {
            "symbol": d["symbol"],
            "price": float(d["lastPrice"]),
            "change_pct": float(d["priceChangePercent"]),
            "volume": float(d["volume"]),
            "high": float(d["highPrice"]),
            "low": float(d["lowPrice"]),
            "provider": "binance",
            "source_mode": SOURCE_LIVE,
            "timestamp": _now_iso(),
        }
    elif provider == "kraken":
        url = f"{KRAKEN_REST}/Ticker"
        r = requests.get(url, params={"pair": symbol}, timeout=3)
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            raise ValueError(f"Kraken error: {data['error']}")
        result = data.get("result", {})
        pair_key = [k for k in result.keys() if k != "last"]
        if not pair_key:
            raise ValueError("No Kraken ticker data")
        d = result[pair_key[0]]
        return {
            "symbol": symbol,
            "price": float(d["c"][0]),
            "change_pct": 0.0,
            "volume": float(d["v"][1]),
            "high": float(d["h"][1]),
            "low": float(d["l"][1]),
            "provider": "kraken",
            "source_mode": SOURCE_LIVE,
            "timestamp": _now_iso(),
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")


def test_provider_connectivity(provider: str = "binance") -> dict:
    """Test connectivity to a provider."""
    if provider == "binance":
        try:
            r = requests.get(f"{BINANCE_REST}/api/v3/ping", timeout=3)
            return {"provider": "binance", "reachable": r.status_code == 200, "latency_ms": r.elapsed.total_seconds() * 1000}
        except Exception as e:
            return {"provider": "binance", "reachable": False, "error": str(e)}
    elif provider == "kraken":
        try:
            r = requests.get(f"{KRAKEN_REST}/Time", timeout=3)
            return {"provider": "kraken", "reachable": r.status_code == 200, "latency_ms": r.elapsed.total_seconds() * 1000}
        except Exception as e:
            return {"provider": "kraken", "reachable": False, "error": str(e)}
    else:
        return {"provider": provider, "reachable": False, "error": "Unknown provider"}


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    print("Testing Binance connectivity:")
    print(test_provider_connectivity("binance"))
    print("Testing Kraken connectivity:")
    print(test_provider_connectivity("kraken"))
