"""Market data provider adapters (Binance REST + WebSocket, Kraken REST)."""
from __future__ import annotations

import asyncio
import fcntl
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
from http_client import http_get

from config import DATA, HISTORY_INTERVALS, FINE_GRAIN_INTERVALS, FINE_GRAIN_UNIVERSE_SIZE, MAX_BACKFILL_CONCURRENCY, MAX_LIVE_CONCURRENCY, SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC
from sources import get_source, approved_sources

log = logging.getLogger(__name__)
RAW = DATA / "raw" / "market"
MARKET = DATA / "market"
SYNTHETIC = DATA / "synthetic"
STATE = DATA / "state"

BINANCE_REST = "https://api.binance.com"
# Official Binance Spot REST failover hosts. The final host is dedicated to
# public market-data endpoints and does not require API credentials.
BINANCE_SPOT_REST_HOSTS = (
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://data-api.binance.vision",
)
BINANCE_FUTURES_REST = "https://fapi.binance.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws"
BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"
KRAKEN_REST = "https://api.kraken.com/0/public"
COINDCX_REST = "https://api.coindcx.com"
_COINDCX_MARKETS_CACHE: list[dict] | None = None
_COINDCX_MARKETS_CACHE_AT = 0.0
_COINDCX_TICKERS_CACHE: dict[str, dict] = {}
_COINDCX_TICKERS_CACHE_AT = 0.0


def _binance_get(path: str, params: dict | None = None, timeout: float = 15,
                 market_type: str = "spot") -> requests.Response:
    """GET public data from Binance, failing over hosts only for transport failures.

    HTTP errors (including 4xx/429) are surfaced immediately rather than masked by
    switching hosts; the caller must handle authorization and rate-limit responses.
    """
    if market_type == "spot":
        hosts = BINANCE_SPOT_REST_HOSTS
    else:
        hosts = (BINANCE_FUTURES_REST,)

    errors = []
    for host in hosts:
        try:
            # Bound total failover latency so a blocked primary host cannot
            # consume the entire collector cycle before another provider is tried.
            return http_get(f"{host}{path}", params=params, timeout=min(float(timeout), 2.5))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.SSLError) as exc:
            errors.append(f"{host}: {type(exc).__name__}: {str(exc)[:180]}")
            log.warning("Binance transport failed at %s; trying next official host", host)

    raise requests.exceptions.ConnectionError(
        "All Binance REST hosts failed: " + " | ".join(errors)
    )


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _add_provenance(df: pd.DataFrame, source_mode: str, provider: str) -> pd.DataFrame:
    """Add missing provenance columns without overwriting verified source labels."""
    df = df.copy()
    if "ingested_at" not in df.columns:
        df["ingested_at"] = _now_iso()
    if "source_mode" not in df.columns:
        df["source_mode"] = source_mode
    if "provider" not in df.columns:
        df["provider"] = provider
    return df


def _dedupe_candles(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Resolve duplicate bars in favor of live exchange rows, then trusted history, then demo."""
    if df.empty or time_col not in df.columns:
        return df
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    priority = {"SYNTHETIC_DEMO": 0, "HISTORICAL_EXCHANGE": 1, "LIVE_EXCHANGE": 2}
    work["_source_priority"] = work.get("source_mode", pd.Series(index=work.index, dtype=str)).map(priority).fillna(-1)
    if "ingested_at" in work.columns:
        work["_ingested_sort"] = pd.to_datetime(work["ingested_at"], utc=True, errors="coerce")
    else:
        work["_ingested_sort"] = pd.NaT
    work = work.dropna(subset=[time_col]).sort_values([time_col, "_source_priority", "_ingested_sort"])
    work = work.drop_duplicates(subset=[time_col], keep="last")
    return work.drop(columns=["_source_priority", "_ingested_sort"], errors="ignore").reset_index(drop=True)


# ---- Binance REST OHLCV ----

def rest_klines(symbol: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None, limit: int = 1000, market_type: str = "spot") -> pd.DataFrame:
    """Fetch klines from Binance REST API. Returns empty DataFrame on failure."""
    path = "/api/v3/klines" if market_type == "spot" else "/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    r = _binance_get(path, params=params, timeout=30, market_type=market_type)
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


def _append_quarantine_frame(frame: pd.DataFrame, path: Path, time_col: str) -> None:
    """Preserve demo/legacy rows away from the production market store."""
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    piece = frame.copy()
    if time_col in piece.columns:
        piece[time_col] = pd.to_datetime(piece[time_col], utc=True, errors="coerce")
    if path.exists():
        try:
            old = pd.read_parquet(path)
            if time_col in old.columns:
                old[time_col] = pd.to_datetime(old[time_col], utc=True, errors="coerce")
            piece = pd.concat([old, piece], ignore_index=True)
        except Exception as exc:
            log.warning("Could not append existing quarantine file %s: %s", path, exc)
    if time_col in piece.columns:
        piece = piece.drop_duplicates(subset=[time_col], keep="last").sort_values(time_col)
    tmp = path.with_suffix(path.suffix + ".tmp")
    piece.to_parquet(tmp, index=False, engine="pyarrow")
    tmp.replace(path)


def save_candles(df: pd.DataFrame, symbol: str, interval: str, market_type: str = "spot", source_mode: str = SOURCE_HISTORICAL, provider: str = "binance"):
    """Atomically persist real candles to production and demo candles to isolated storage."""
    lock_root = (SYNTHETIC / "raw" / "market") if source_mode == SOURCE_SYNTHETIC else RAW
    lock_dir = lock_root / market_type / symbol
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{interval}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _save_candles_unlocked(df, symbol, interval, market_type, source_mode, provider)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _save_candles_unlocked(df: pd.DataFrame, symbol: str, interval: str, market_type: str = "spot", source_mode: str = SOURCE_HISTORICAL, provider: str = "binance"):
    """Write a single provenance class under the matching storage root."""
    if df.empty:
        return 0

    df = _add_provenance(df, source_mode, provider)
    if "open_time" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "open_time"})
    if "open_time" not in df.columns:
        raise ValueError("Cannot save candles without open_time/timestamp provenance")

    modes = set(df["source_mode"].dropna().astype(str).unique()) if "source_mode" in df.columns else set()
    allowed_modes = {SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC}
    if len(modes) != 1 or not modes.issubset(allowed_modes):
        raise ValueError(f"Refusing candle batch with ambiguous or mixed provenance: {sorted(modes)}")
    mode = next(iter(modes))
    is_demo = mode == SOURCE_SYNTHETIC
    raw_root = SYNTHETIC / "raw" / "market" if is_demo else RAW
    norm_root = SYNTHETIC / "market" if is_demo else MARKET

    # Reject invalid incoming bars before deduplication so a bad real row cannot
    # delete a previously stored valid row at the same timestamp.
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    invalid_input = (
        df[list(("open", "high", "low", "close", "volume"))].isna().any(axis=1)
        | (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
        | (df["volume"] < 0)
        | (df["high"] < df["low"])
        | (df["high"] < df["open"]) | (df["high"] < df["close"])
        | (df["low"] > df["open"]) | (df["low"] > df["close"])
    )
    if invalid_input.any():
        log.warning("Dropping %d invalid incoming OHLC rows for %s %s", int(invalid_input.sum()), symbol, interval)
        df = df.loc[~invalid_input].copy()
    if df.empty:
        return 0

    raw_dir = raw_root / market_type / symbol
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

    # Older builds wrote demo rows into the production path. Move unverified rows
    # into the isolated synthetic/legacy archive as soon as a real batch touches
    # the series. This also prevents later writes from resurrecting demo candles.
    if not is_demo and not existing.empty:
        real_mode_mask = existing.get("source_mode", pd.Series(index=existing.index, dtype=str)).astype(str).isin([SOURCE_LIVE, SOURCE_HISTORICAL])
        non_real = existing.loc[~real_mode_mask].copy()
        if not non_real.empty:
            target = (SYNTHETIC / "raw" / "market" / market_type / symbol / f"{interval}.parquet")
            _append_quarantine_frame(non_real, target, "open_time")
            existing = existing.loc[real_mode_mask].copy()

    existing_before = existing.copy()
    combined = _dedupe_candles(pd.concat([existing, df], ignore_index=True), "open_time")

    bad = (combined["high"] < combined["low"]) | (combined["high"] < combined["open"]) | (combined["high"] < combined["close"]) | (combined["low"] > combined["open"]) | (combined["low"] > combined["close"])
    if bad.any():
        log.warning("Dropping %d invalid OHLC rows for %s %s", bad.sum(), symbol, interval)
        combined = combined[~bad]

    # Count new observations and provenance upgrades, not just net file growth.
    # A genuine candle replacing a synthetic candle is useful progress even if
    # the total row count stays constant.
    valid_keys = set(pd.to_datetime(combined["open_time"], utc=True, errors="coerce").dropna())
    old_priorities = {}
    if not existing_before.empty and "open_time" in existing_before.columns:
        old_before = existing_before.copy()
        old_before["open_time"] = pd.to_datetime(old_before["open_time"], utc=True, errors="coerce")
        priority_map = {"SYNTHETIC_DEMO": 0, "HISTORICAL_EXCHANGE": 1, "LIVE_EXCHANGE": 2}
        old_before["_priority"] = old_before.get("source_mode", pd.Series(index=old_before.index, dtype=str)).map(priority_map).fillna(-1)
        old_priorities = dict(zip(old_before["open_time"], old_before["_priority"]))

    new_input = df.copy()
    new_input["open_time"] = pd.to_datetime(new_input["open_time"], utc=True, errors="coerce")
    priority_map = {"SYNTHETIC_DEMO": 0, "HISTORICAL_EXCHANGE": 1, "LIVE_EXCHANGE": 2}
    new_input["_priority"] = new_input.get("source_mode", pd.Series(index=new_input.index, dtype=str)).map(priority_map).fillna(-1)
    new_input = new_input[new_input["open_time"].isin(valid_keys)].sort_values(["open_time", "_priority"])
    new_input = new_input.drop_duplicates(subset=["open_time"], keep="last")
    written = sum(
        1 for _, item in new_input.iterrows()
        if item["open_time"] not in old_priorities or item["_priority"] > old_priorities[item["open_time"]]
    )
    
    tmp_raw = raw_path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp_raw, index=False, engine="pyarrow")
    tmp_raw.replace(raw_path)

    norm_dir = norm_root / market_type / symbol
    norm_dir.mkdir(parents=True, exist_ok=True)
    norm_path = norm_dir / f"{interval}.parquet"
    
    norm_df = combined[["open_time", "open", "high", "low", "close", "volume", "symbol", "interval", "market", "exchange", "ingested_at", "source_mode", "provider"]].copy()
    norm_df.columns = ["timestamp", "open", "high", "low", "close", "volume", "symbol", "interval", "market", "exchange", "ingested_at", "source_mode", "provider"]
    if norm_path.exists():
        try:
            old = pd.read_parquet(norm_path)
            old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
            if not is_demo:
                real_mask = old.get("source_mode", pd.Series(index=old.index, dtype=str)).astype(str).isin([SOURCE_LIVE, SOURCE_HISTORICAL])
                non_real = old.loc[~real_mask].copy()
                if not non_real.empty:
                    target = SYNTHETIC / "market" / market_type / symbol / f"{interval}.parquet"
                    _append_quarantine_frame(non_real, target, "timestamp")
                    old = old.loc[real_mask].copy()
            norm_df = _dedupe_candles(pd.concat([old, norm_df], ignore_index=True), "timestamp")
        except Exception:
            pass
    tmp_norm = norm_path.with_suffix(".parquet.tmp")
    norm_df.to_parquet(tmp_norm, index=False, engine="pyarrow")
    tmp_norm.replace(norm_path)
    
    return written


def _utc_timestamp(value) -> pd.Timestamp:
    """Normalize naive or aware timestamps safely to UTC."""
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def get_candles(symbol: str, interval: str, market_type: str = "spot", start: Optional[datetime] = None, end: Optional[datetime] = None, source_mode: Optional[str] = None) -> pd.DataFrame:
    """Read production exchange candles by default; demo candles require an explicit source mode."""
    if source_mode is not None and source_mode not in {SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC}:
        return pd.DataFrame()
    path_root = SYNTHETIC / "market" if source_mode == SOURCE_SYNTHETIC else MARKET
    path = path_root / market_type / symbol / f"{interval}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "source_mode" not in df.columns:
        return pd.DataFrame()
    if source_mode is None:
        # Default callers, including APIs, must only see exchange-derived data.
        df = df[df["source_mode"].isin([SOURCE_LIVE, SOURCE_HISTORICAL])]
    else:
        df = df[df["source_mode"] == source_mode]
    if start:
        df = df[df["timestamp"] >= _utc_timestamp(start)]
    if end:
        df = df[df["timestamp"] <= _utc_timestamp(end)]
    return df.sort_values("timestamp").reset_index(drop=True)


def get_last_timestamp(symbol: str, interval: str, market_type: str = "spot", source_mode: Optional[str] = None) -> Optional[datetime]:
    df = get_candles(symbol, interval, market_type, source_mode=source_mode)
    if df.empty:
        return None
    return df["timestamp"].max().to_pydatetime()


# ---- Kraken REST OHLCV ----

_KRAKEN_PAIRS_CACHE: list[dict] | None = None
_KRAKEN_PAIRS_CACHE_AT = 0.0


def _normalise_asset(value: str) -> str:
    value = str(value).upper().strip()
    return "BTC" if value == "XBT" else value


def _symbol_parts(symbol: str) -> tuple[str, str]:
    clean = str(symbol).upper().replace("/", "").replace("-", "")
    for quote in ("USDT", "USDC", "USD", "EUR", "GBP"):
        if clean.endswith(quote) and len(clean) > len(quote):
            return _normalise_asset(clean[:-len(quote)]), _normalise_asset(quote)
    raise ValueError(f"Unsupported Kraken symbol format: {symbol}")


def kraken_asset_pairs(refresh: bool = False) -> list[dict]:
    """Return Kraken's current pair metadata with a short in-process cache."""
    global _KRAKEN_PAIRS_CACHE, _KRAKEN_PAIRS_CACHE_AT
    if not refresh and _KRAKEN_PAIRS_CACHE is not None and time.time() - _KRAKEN_PAIRS_CACHE_AT < 900:
        return _KRAKEN_PAIRS_CACHE
    response = http_get(f"{KRAKEN_REST}/AssetPairs", timeout=15)
    data = response.json()
    if data.get("error"):
        raise ValueError(f"Kraken AssetPairs error: {data['error']}")
    result = data.get("result", {})
    _KRAKEN_PAIRS_CACHE = [dict(pair_key=key, **value) for key, value in result.items() if isinstance(value, dict)]
    _KRAKEN_PAIRS_CACHE_AT = time.time()
    return _KRAKEN_PAIRS_CACHE


def resolve_kraken_pair(symbol: str) -> str:
    """Resolve an internal symbol to an exact Kraken pair; never silently change quote currency."""
    if "/" in str(symbol):
        wanted_base, wanted_quote = (_normalise_asset(x) for x in str(symbol).split("/", 1))
    else:
        wanted_base, wanted_quote = _symbol_parts(symbol)
    for pair in kraken_asset_pairs():
        wsname = str(pair.get("wsname", ""))
        if "/" not in wsname:
            continue
        base, quote = (_normalise_asset(x) for x in wsname.split("/", 1))
        if base == wanted_base and quote == wanted_quote and pair.get("status", "online") == "online":
            return str(pair.get("altname") or pair.get("pair_key"))
    raise ValueError(f"Kraken has no active exact-quote pair for {symbol}")


def kraken_ohlc(pair: str, interval: int = 60, since: Optional[int] = None,
                symbol: Optional[str] = None, interval_label: Optional[str] = None,
                source_mode: str = SOURCE_HISTORICAL) -> pd.DataFrame:
    """Fetch completed Kraken OHLC candles for an exact exchange pair.

    Kraken's OHLC endpoint returns at most 720 recent rows and includes the
    currently forming candle; discard the unfinished row and report actual coverage.
    """
    resolved_pair = resolve_kraken_pair(pair)
    url = f"{KRAKEN_REST}/OHLC"
    params = {"pair": resolved_pair, "interval": interval}
    if since is not None:
        params["since"] = since

    response = http_get(url, params=params, timeout=30)
    data = response.json()
    if data.get("error"):
        raise ValueError(f"Kraken error: {data['error']}")

    result = data.get("result", {})
    pair_keys = [key for key in result.keys() if key != "last"]
    if not pair_keys:
        return pd.DataFrame()

    raw = result[pair_keys[0]]
    cols = ["open_time", "open", "high", "low", "close", "vwap", "volume", "count"]
    df = pd.DataFrame(raw, columns=cols)
    if df.empty:
        return df
    for col in ["open", "high", "low", "close", "volume", "vwap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="s", utc=True)
    df["close_time"] = df["open_time"] + pd.to_timedelta(interval, unit="m")
    df = df[df["close_time"] <= pd.Timestamp.now(tz="UTC")].copy()
    if df.empty:
        return df
    df["symbol"] = str(symbol or pair).replace("/", "").replace("-", "").upper()
    inferred = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h",
                240: "4h", 1440: "1d", 10080: "7d", 21600: "15d"}.get(interval, f"{interval}m")
    df["interval"] = interval_label or inferred
    df["market"] = "spot"
    df["exchange"] = "kraken"
    df["trades"] = pd.to_numeric(df["count"], errors="coerce")
    df["ingested_at"] = _now_iso()
    df["source_mode"] = source_mode
    df["provider"] = "kraken"
    return df


# ---- CoinDCX public spot market data (no account credentials required) ----

def coindcx_market_details(refresh: bool = False) -> list[dict]:
    """Fetch active CoinDCX public market metadata; never uses account credentials."""
    global _COINDCX_MARKETS_CACHE, _COINDCX_MARKETS_CACHE_AT
    if not refresh and _COINDCX_MARKETS_CACHE is not None and time.time() - _COINDCX_MARKETS_CACHE_AT < 900:
        return _COINDCX_MARKETS_CACHE
    response = http_get(f"{COINDCX_REST}/exchange/v1/markets_details", timeout=12)
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected CoinDCX markets-details response")
    _COINDCX_MARKETS_CACHE = [item for item in payload if isinstance(item, dict)]
    _COINDCX_MARKETS_CACHE_AT = time.time()
    return _COINDCX_MARKETS_CACHE


def resolve_coindcx_pair(symbol: str) -> str:
    """Resolve canonical symbol like BTCUSDT to an active exact CoinDCX pair."""
    wanted = str(symbol).upper().replace("/", "").replace("-", "")
    for item in coindcx_market_details():
        if str(item.get("status", "active")).lower() != "active":
            continue
        candidate = str(item.get("symbol") or item.get("coindcx_name") or "").upper().replace("/", "").replace("-", "")
        if candidate == wanted and item.get("pair"):
            return str(item["pair"])
    raise ValueError(f"CoinDCX has no active exact market for {symbol}")


def discover_coindcx_universe(quote_asset: str = "USDT", limit: int = 200) -> pd.DataFrame:
    """Discover active CoinDCX spot pairs via unauthenticated public market metadata."""
    rows = []
    for item in coindcx_market_details():
        if str(item.get("status", "")).lower() != "active":
            continue
        base = str(item.get("target_currency_short_name", "")).upper()
        quote = str(item.get("base_currency_short_name", "")).upper()
        symbol = str(item.get("symbol") or item.get("coindcx_name") or "").upper()
        pair = str(item.get("pair", ""))
        if not base or not quote or quote != quote_asset.upper() or not pair:
            continue
        if symbol != f"{base}{quote}":
            continue
        rows.append({
            "symbol": symbol, "base_asset": base, "quote_asset": quote,
            "status": "TRADING", "contract_type": "SPOT", "exchange": "coindcx",
            "market": "spot", "exchange_pair": pair,
            "universe_source": "COINDCX_MARKETS_DETAILS",
            "universe_provenance": SOURCE_HISTORICAL, "priority": 0,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["symbol"]).sort_values(["priority", "base_asset"]).head(limit).reset_index(drop=True)
    return df


def coindcx_ohlc(symbol: str, interval: str = "1h",
                  start_ms: int | None = None, end_ms: int | None = None,
                  limit: int = 1000, source_mode: str = SOURCE_HISTORICAL) -> pd.DataFrame:
    """Fetch completed CoinDCX spot candles using its public, unauthenticated endpoint."""
    if interval not in {"1m", "15m", "1h", "1d"}:
        raise ValueError(f"Unsupported CoinDCX interval: {interval}")
    pair = resolve_coindcx_pair(symbol)
    params = {"pair": pair, "interval": interval, "limit": min(max(int(limit), 1), 1000)}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    response = http_get(f"{COINDCX_REST}/market_data/candles", params=params, timeout=15)
    raw = response.json()
    if not isinstance(raw, list) or not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    required = {"time", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError("Unexpected CoinDCX candle schema")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["time"], errors="coerce"), unit="ms", utc=True)
    duration = {"1m": pd.Timedelta(minutes=1), "15m": pd.Timedelta(minutes=15),
                "1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1)}[interval]
    df["close_time"] = df["open_time"] + duration
    df = df.dropna(subset=["open_time", "open", "high", "low", "close", "volume"])
    df = df[df["close_time"] <= pd.Timestamp.now(tz="UTC")].copy()
    if df.empty:
        return df
    df["symbol"] = str(symbol).upper().replace("/", "").replace("-", "")
    df["interval"] = interval
    df["market"] = "spot"
    df["exchange"] = "coindcx"
    df["ingested_at"] = _now_iso()
    df["source_mode"] = source_mode
    df["provider"] = "coindcx"
    return df.sort_values("open_time").reset_index(drop=True)


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
    
    # Synthetic timestamps must never make a real-provider backfill look complete.
    existing_real = get_candles(symbol, interval, market_type, source_mode=SOURCE_HISTORICAL)
    if not existing_real.empty and "provider" in existing_real.columns:
        existing_real = existing_real[existing_real["provider"].astype(str).str.lower() == provider.lower()]
    existing_end = existing_real["timestamp"].max().to_pydatetime() if not existing_real.empty else None
    freshness_hours = 26 if interval == "1d" else 2
    if existing_end and existing_end >= end - pd.Timedelta(hours=freshness_hours):
        log.info("Backfill for %s %s %s already complete through %s", symbol, interval, market_type, existing_end)
        return total_rows
    
    cur = max(current_start, start)
    if provider == "kraken":
        # Kraken OHLC returns at most 720 bars and cannot page into older history.
        interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
                            "1h": 60, "4h": 240, "1d": 1440, "7d": 10080}
        if interval not in interval_minutes:
            raise ValueError(f"Unsupported Kraken interval: {interval}")
        cur = max(start, end - pd.Timedelta(minutes=interval_minutes[interval] * 720))
    batch = 1000
    consecutive_errors = 0
    
    while cur < end:
        start_ms = int(cur.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        
        try:
            if provider == "binance":
                df = rest_klines(symbol, interval, start_ms=start_ms, end_ms=end_ms, limit=batch, market_type=market_type)
            elif provider == "kraken":
                # Kraken OHLC supports only the latest 720 candles per interval.
                interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
                                    "1h": 60, "4h": 240, "1d": 1440, "7d": 10080}
                if interval not in interval_minutes:
                    raise ValueError(f"Unsupported Kraken candle interval: {interval}")
                kraken_interval = interval_minutes[interval]
                effective_start = max(cur, end - pd.Timedelta(minutes=kraken_interval * 720))
                df = kraken_ohlc(symbol, interval=kraken_interval,
                                  since=int(effective_start.timestamp()),
                                  symbol=symbol, interval_label=interval)
                if not df.empty:
                    df = df[(df["open_time"] >= pd.Timestamp(effective_start)) &
                            (df["open_time"] <= pd.Timestamp(end))]
            elif provider == "coindcx":
                step_ms = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}
                if interval not in step_ms:
                    raise ValueError(f"Unsupported CoinDCX candle interval: {interval}")
                chunk_end = min(end, cur + pd.Timedelta(milliseconds=step_ms[interval] * (batch - 1)))
                df = coindcx_ohlc(symbol, interval=interval, start_ms=start_ms,
                                  end_ms=int(chunk_end.timestamp() * 1000), limit=batch)
                if not df.empty:
                    df = df[(df["open_time"] >= pd.Timestamp(cur)) &
                            (df["open_time"] <= pd.Timestamp(chunk_end))]
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            if df.empty:
                break
            
            written = save_candles(df, symbol, interval, market_type, source_mode=SOURCE_HISTORICAL, provider=provider)
            total_rows += written
            new_end = df["open_time"].max().to_pydatetime() if "open_time" in df.columns else df["time"].max().to_pydatetime()
            if provider == "coindcx":
                step_ms = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}[interval]
                cur = new_end + pd.Timedelta(milliseconds=step_ms)
            else:
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
            if provider == "kraken":
                # Kraken only returns the newest 720 bars: a second page call would
                # repeat the same window and can loop forever.
                break
            
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
    # Binance combined-stream syntax differs from the single-stream /ws endpoint.
    if len(streams) == 1:
        base = BINANCE_WS if market_type == "spot" else BINANCE_FUTURES_WS
        url = f"{base}/{streams[0]}"
    else:
        base = "wss://stream.binance.com:9443/stream?streams=" if market_type == "spot" else "wss://fstream.binance.com/stream?streams="
        url = base + "/".join(streams)
    backoff = 1
    while True:
        try:
            log.info("Connecting WS streams=%d provider=binance", len(streams))
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=3) as ws:
                backoff = 1
                last_msg = time.time()
                async for msg in ws:
                    data = json.loads(msg)
                    # Combined streams wrap the exchange event in {stream, data}.
                    data = data.get("data", data)
                    last_msg = time.time()
                    if data.get("e") == "kline":
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
    """Fetch 24-hour ticker metadata from the named provider or first approved reachable provider."""
    if provider == "auto":
        errors = []
        for candidate in ("binance", "kraken", "coindcx"):
            try:
                return fetch_ticker(symbol, market_type, candidate)
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
        raise RuntimeError("No ticker provider available: " + "; ".join(errors))

    if provider == "binance":
        path = "/api/v3/ticker/24hr" if market_type == "spot" else "/fapi/v1/ticker/24hr"
        response = _binance_get(path, params={"symbol": symbol}, timeout=8, market_type=market_type)
        data = response.json()
        return {
            "symbol": data["symbol"],
            "price": float(data["lastPrice"]),
            "change_pct": float(data["priceChangePercent"]),
            "volume": float(data["volume"]),
            "high": float(data["highPrice"]),
            "low": float(data["lowPrice"]),
            "provider": "binance",
            "source_mode": SOURCE_LIVE,
            "timestamp": _now_iso(),
            "transport": response.headers.get("X-NeuroTrom-Transport", "requests"),
        }

    if provider == "kraken":
        if market_type != "spot":
            raise ValueError("Kraken adapter in this release supports spot market data only")
        exact_pair = resolve_kraken_pair(symbol)
        response = http_get(f"{KRAKEN_REST}/Ticker", params={"pair": exact_pair}, timeout=8)
        payload = response.json()
        if payload.get("error"):
            raise ValueError(f"Kraken error: {payload['error']}")
        result = payload.get("result", {})
        pair_key = [key for key in result if key != "last"]
        if not pair_key:
            raise ValueError(f"No Kraken ticker result for {symbol}")
        data = result[pair_key[0]]
        return {
            "symbol": symbol,
            "price": float(data["c"][0]),
            "change_pct": None,  # Kraken's "o" is not guaranteed to be a rolling 24h open.
            "volume": float(data["v"][1]),
            "high": float(data["h"][1]),
            "low": float(data["l"][1]),
            "provider": "kraken",
            "source_mode": SOURCE_LIVE,
            "timestamp": _now_iso(),
            "transport": response.headers.get("X-NeuroTrom-Transport", "requests"),
        }

    if provider == "coindcx":
        if market_type != "spot":
            raise ValueError("CoinDCX adapter supports public spot market data only")
        global _COINDCX_TICKERS_CACHE, _COINDCX_TICKERS_CACHE_AT
        now = time.monotonic()
        if not _COINDCX_TICKERS_CACHE or now - _COINDCX_TICKERS_CACHE_AT > 1.5:
            response = http_get(f"{COINDCX_REST}/exchange/ticker", timeout=10)
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Unexpected CoinDCX ticker response")
            _COINDCX_TICKERS_CACHE = {
                str(item.get("market", "")).upper(): item
                for item in payload if isinstance(item, dict) and item.get("market")
            }
            _COINDCX_TICKERS_CACHE_AT = now
        key = str(symbol).upper().replace("/", "").replace("-", "")
        item = _COINDCX_TICKERS_CACHE.get(key)
        if item is None:
            raise ValueError(f"CoinDCX ticker has no exact market for {symbol}")
        try:
            stamp = pd.to_datetime(int(item.get("timestamp")), unit="ms", utc=True).isoformat()
        except (TypeError, ValueError, OverflowError):
            stamp = _now_iso()
        return {
            "symbol": key,
            "price": float(item["last_price"]),
            "change_pct": float(item["change_24_hour"]) if item.get("change_24_hour") is not None else None,
            "volume": float(item["volume"]) if item.get("volume") is not None else None,
            "high": float(item["high"]) if item.get("high") is not None else None,
            "low": float(item["low"]) if item.get("low") is not None else None,
            "provider": "coindcx",
            "source_mode": SOURCE_LIVE,
            "timestamp": stamp,
            "transport": response.headers.get("X-NeuroTrom-Transport", "requests") if "response" in locals() else "ticker-cache",
        }

    raise ValueError(f"Unknown provider: {provider}")

def test_provider_connectivity(provider: str = "binance") -> dict:
    """Test connectivity to a provider."""
    if provider == "binance":
        errors = []
        for host in BINANCE_SPOT_REST_HOSTS:
            try:
                t0 = time.monotonic()
                r = http_get(f"{host}/api/v3/ping", timeout=2)
                return {
                    "provider": "binance",
                    "reachable": r.status_code == 200,
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    "transport": r.headers.get("X-NeuroTrom-Transport", "requests"),
                    "host": host,
                }
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.SSLError) as e:
                errors.append(f"{host}: {type(e).__name__}: {str(e)[:160]}")
                continue
            except Exception as e:
                # Non-transport errors should remain visible, not be hidden by failover.
                return {"provider": "binance", "reachable": False, "host": host, "error": str(e)}
        return {"provider": "binance", "reachable": False, "error": "All Binance hosts failed: " + " | ".join(errors)}
    elif provider == "kraken":
        try:
            t0 = time.monotonic()
            r = http_get(f"{KRAKEN_REST}/Time", timeout=4)
            data = r.json()
            if data.get("error"):
                raise ValueError(f"Kraken error: {data['error']}")
            return {"provider": "kraken", "reachable": r.status_code == 200, "latency_ms": round((time.monotonic() - t0) * 1000, 1), "transport": r.headers.get("X-NeuroTrom-Transport", "requests")}
        except Exception as e:
            return {"provider": "kraken", "reachable": False, "error": str(e)}
    elif provider == "coindcx":
        try:
            t0 = time.monotonic()
            r = http_get(f"{COINDCX_REST}/exchange/ticker", timeout=8)
            data = r.json()
            reachable = isinstance(data, list) and len(data) > 0
            return {"provider": "coindcx", "reachable": reachable, "latency_ms": round((time.monotonic() - t0) * 1000, 1), "transport": r.headers.get("X-NeuroTrom-Transport", "requests")}
        except Exception as e:
            return {"provider": "coindcx", "reachable": False, "error": str(e)}
    else:
        return {"provider": provider, "reachable": False, "error": "Unknown provider"}


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    print("Testing Binance connectivity:")
    print(test_provider_connectivity("binance"))
    print("Testing Kraken connectivity:")
    print(test_provider_connectivity("kraken"))
