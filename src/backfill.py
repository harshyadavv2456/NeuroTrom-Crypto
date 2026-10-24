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
    """Backfill real exchange-derived symbols using their recorded exchange."""
    if intervals is None:
        intervals = HISTORY_INTERVALS

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    df = load_universe()
    if df.empty:
        df = build_universe(universe_limit)
    if df.empty or "symbol" not in df.columns:
        return 0, ["No universe available"]

    state = load_backfill_state()
    total = 0
    errors = []
    tasks = []

    # Schedule provider-specific work. Demo symbols are deliberately not backfilled.
    with ThreadPoolExecutor(max_workers=MAX_BACKFILL_CONCURRENCY) as pool:
        futures = {}
        for _, record in df.head(universe_limit).iterrows():
            sym = str(record["symbol"])
            provider = str(record.get("exchange", "binance")).lower()
            if provider not in {"binance", "kraken"}:
                errors.append(f"{sym}: skipped; universe provenance is demo/unknown")
                continue

            for interval in intervals:
                key = f"{sym}_{interval}_spot_{provider}"
                prev = state.get("symbols_done", {}).get(key, {})
                try:
                    prev_end = datetime.fromisoformat(prev["end"]) if prev.get("end") else None
                    if prev_end and prev_end.tzinfo is None:
                        prev_end = prev_end.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    prev_end = None
                # Never let an earlier zero-row attempt permanently mark a symbol done.
                if prev.get("rows", 0) > 0 and prev_end and prev_end >= end - timedelta(hours=2):
                    continue

                future = pool.submit(backfill_symbol, sym, interval, start, end, "spot", 100, provider)
                futures[future] = (sym, interval, provider, key)

        for future in as_completed(futures):
            sym, interval, provider, key = futures[future]
            try:
                rows = int(future.result() or 0)
                total += rows
                if rows > 0:
                    state.setdefault("symbols_done", {})[key] = {
                        "end": end.isoformat(),
                        "rows": rows,
                        "provider": provider,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    errors.append(f"{sym} {interval} via {provider}: no new candles returned")
            except Exception as exc:
                errors.append(f"{sym} {interval} via {provider}: {exc}")
                log.error("Backfill error %s %s via %s: %s", sym, interval, provider, exc)

    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["symbols_done"] = state.get("symbols_done", {})
    state["errors"] = errors[-100:]
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
    """Report real exchange quality separately from demo-only data."""
    all_df = get_candles(symbol, interval, market_type)
    if all_df.empty:
        return {
            "status": "empty",
            "rows": 0,
            "real_rows": 0,
            "synthetic_rows": 0,
            "source_mode": "NONE",
            "note": "No stored candles",
        }

    if "source_mode" not in all_df.columns:
        real_df = pd.DataFrame()
        synthetic_rows = 0
        unknown_rows = len(all_df)
    else:
        modes = all_df["source_mode"].astype(str)
        real_mask = modes.isin(["LIVE_EXCHANGE", "HISTORICAL_EXCHANGE"])
        synthetic_rows = int(modes.eq("SYNTHETIC_DEMO").sum())
        unknown_rows = int((~real_mask & ~modes.eq("SYNTHETIC_DEMO")).sum())
        real_df = all_df.loc[real_mask].copy()

    if real_df.empty:
        status = "demo_only" if synthetic_rows else "unknown_provenance"
        return {
            "status": status,
            "rows": 0,
            "real_rows": 0,
            "synthetic_rows": synthetic_rows,
            "unknown_rows": unknown_rows,
            "first": None,
            "last": None,
            "age_hours": None,
            "gap_pct": None,
            "market": market_type,
            "source_mode": "SYNTHETIC_DEMO" if synthetic_rows else "UNKNOWN",
            "note": "No verified exchange candles; demo data is excluded from real-data quality.",
        }

    real_df["timestamp"] = pd.to_datetime(real_df["timestamp"], utc=True, errors="coerce")
    real_df = real_df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    now = datetime.now(timezone.utc)
    first = real_df["timestamp"].min().to_pydatetime()
    last = real_df["timestamp"].max().to_pydatetime()
    age_hours = max(0.0, (now - last).total_seconds() / 3600)

    expected = {"1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
                "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30),
                "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4),
                "1d": pd.Timedelta(days=1), "7d": pd.Timedelta(days=7)}.get(interval)
    if expected and len(real_df) > 1:
        expected_rows = int((last - first) / expected) + 1
        missing_rows = max(0, expected_rows - int(real_df["timestamp"].nunique()))
        gap_pct = 100.0 * missing_rows / max(1, expected_rows)
    else:
        gap_pct = 0.0 if len(real_df) else None

    modes = set(real_df["source_mode"].astype(str).unique()) if "source_mode" in real_df.columns else {"UNKNOWN"}
    providers = sorted(set(real_df["provider"].astype(str).unique())) if "provider" in real_df.columns else ["unknown"]
    freshness_limit = 3.5 if interval in ("1m", "5m", "15m", "30m", "1h", "4h") else 30.0
    if age_hours > freshness_limit:
        status = "stale"
    elif gap_pct is not None and gap_pct > 5:
        status = "gaps"
    else:
        status = "ok"

    source_label = next(iter(modes)) if len(modes) == 1 else "REAL_EXCHANGE_MIXED"
    return {
        "status": status,
        "rows": int(len(real_df)),
        "real_rows": int(len(real_df)),
        "synthetic_rows": synthetic_rows,
        "unknown_rows": unknown_rows,
        "first": first.isoformat(),
        "last": last.isoformat(),
        "age_hours": round(age_hours, 2),
        "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
        "market": market_type,
        "source_mode": source_label,
        "providers": providers,
        "note": "Real exchange observations only; synthetic rows excluded.",
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

