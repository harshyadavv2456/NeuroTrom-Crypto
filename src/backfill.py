"""Resumable historical backfill engine."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config import DATA, HISTORY_INTERVALS, CANDLE_RETENTION_DAYS, MAX_BACKFILL_CONCURRENCY
from universe import load_universe, build_universe
from market_data import backfill_symbol, get_candles, save_candles

log = logging.getLogger(__name__)
STATE = DATA / "state"
BACKFILL_STATE = STATE / "backfill_state.json"


def load_backfill_state() -> dict:
    if BACKFILL_STATE.exists():
        try:
            return json.loads(BACKFILL_STATE.read_text())
        except Exception:
            pass
    return {"last_run": None, "symbols_done": {}, "errors": []}


def save_backfill_state(state: dict):
    BACKFILL_STATE.write_text(json.dumps(state, indent=2, default=str))


def backfill_universe(intervals: list[str] = None, universe_limit: int = 60, days: int = 365):
    """Backfill all symbols in the universe for given intervals."""
    if intervals is None:
        intervals = HISTORY_INTERVALS
    
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    
    df = load_universe()
    if df.empty:
        df = build_universe(universe_limit)
    
    symbols = df["symbol"].tolist()[:universe_limit]
    state = load_backfill_state()
    
    total = 0
    errors = []
    
    # Backfill spot data
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_BACKFILL_CONCURRENCY) as pool:
        futures = {}
        for sym in symbols:
            for interval in intervals:
                key = f"{sym}_{interval}_spot"
                if key in state.get("symbols_done", {}):
                    prev_end = datetime.fromisoformat(state["symbols_done"][key]["end"])
                    if prev_end >= end - timedelta(hours=2):
                        continue
                
                fut = pool.submit(backfill_symbol, sym, interval, start, end, "spot")
                futures[fut] = (sym, interval, "spot")
        
        for fut in as_completed(futures):
            sym, interval, mkt = futures[fut]
            try:
                rows = fut.result()
                total += rows
                key = f"{sym}_{interval}_{mkt}"
                state.setdefault("symbols_done", {})[key] = {
                    "end": end.isoformat(),
                    "rows": rows,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                errors.append(f"{sym} {interval}: {e}")
                log.error("Backfill error %s %s: %s", sym, interval, e)
    
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["errors"] = errors[-20:]
    save_backfill_state(state)
    
    log.info("Backfill complete: %d rows, %d errors", total, len(errors))
    return total, errors


def backfill_fine_grain(symbols: list[str] = None, days: int = 30):
    """Backfill 1-minute data for a small set of symbols."""
    if symbols is None:
        df = load_universe()
        symbols = df["symbol"].tolist()[:15] if not df.empty else ["BTCUSDT", "ETHUSDT"]
    
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    total = 0
    for sym in symbols:
        try:
            rows = backfill_symbol(sym, "1m", start, end, "spot")
            total += rows
        except Exception as e:
            log.error("Fine-grain backfill error %s: %s", sym, e)
    return total


def verify_backfill(symbol: str, interval: str, market_type: str = "spot") -> dict:
    """Check backfill status for one symbol."""
    df = get_candles(symbol, interval, market_type)
    if df.empty:
        return {"status": "empty", "rows": 0}
    
    now = datetime.now(timezone.utc)
    last = df["timestamp"].max().to_pydatetime()
    age_hours = (now - last).total_seconds() / 3600
    
    # Check for gaps
    if len(df) > 1:
        diffs = df["timestamp"].diff().dropna()
        expected = pd.Timedelta(hours=1) if interval == "1h" else pd.Timedelta(days=1) if interval == "1d" else pd.Timedelta(minutes=1)
        gap_threshold = expected * 3
        gaps = diffs[diffs > gap_threshold]
        gap_pct = len(gaps) / len(diffs) * 100
    else:
        gap_pct = 0
    
    source = df["source_mode"].iloc[0] if "source_mode" in df.columns and not df.empty else "unknown"
    return {
        "status": "ok" if age_hours < 3 else "stale",
        "rows": len(df),
        "first": str(df["timestamp"].min()),
        "last": str(last),
        "age_hours": round(age_hours, 1),
        "gap_pct": round(gap_pct, 2),
        "market": market_type,
        "source_mode": source,
    }



def reconcile_gaps():
    """Reconcile gaps in stored data by backfilling missing intervals."""
    from universe import load_universe
    from market_data import get_candles, rest_klines, save_candles
    from config import SOURCE_HISTORICAL
    
    df = load_universe()
    if df.empty:
        return {"reconciled": 0, "errors": ["no universe"]}
    
    gaps_found = 0
    for _, row in df.head(20).iterrows():
        for interval in ["1h", "1d"]:
            result = verify_backfill(row["symbol"], interval, row.get("market", "spot"))
            if result.get("status") == "degraded" or result.get("gap_pct", 0) > 5:
                gaps_found += 1
                try:
                    raw = rest_klines(row["symbol"], interval, limit=500)
                    if not raw.empty:
                        save_candles(raw, row["symbol"], interval, row.get("market", "spot"), source_mode=SOURCE_HISTORICAL, provider="binance")
                except Exception:
                    pass
    
    return {"reconciled": gaps_found, "status": "complete"}

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    r, e = backfill_universe(days=7)
    print(f"Backfill: {r} rows, {len(e)} errors")

