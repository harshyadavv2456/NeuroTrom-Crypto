"""NeuroTrom Crypto local API service (127.0.0.1:8775)."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import DATA, HOST, PORT, HORIZONS, SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC
from market_data import get_candles, rest_klines, save_candles, fetch_ticker, test_provider_connectivity
from universe import load_universe, build_universe
from backfill import backfill_symbol, verify_backfill, backfill_universe, reconcile_gaps, load_backfill_state
from features import build_feature_dataset, load_features
from train import run_training, predict, _check_provenance, MODELS
from journal import compute_journal_summary, record_prediction, mature_predictions
from live import get_live_status, start_live, stop_live
from sources import load_sources, approved_sources

log = logging.getLogger(__name__)
app = FastAPI(title="NeuroTrom Crypto", version="0.1.0")


_PROVIDER_CACHE: dict | None = None
_PROVIDER_CACHE_AT = 0.0
_PROVIDER_CACHE_TTL = 60.0


def _provider_status(refresh: bool = False) -> dict:
    """Cache slow public-provider probes across dashboard/status requests."""
    global _PROVIDER_CACHE, _PROVIDER_CACHE_AT
    if not refresh and _PROVIDER_CACHE is not None and time.monotonic() - _PROVIDER_CACHE_AT < _PROVIDER_CACHE_TTL:
        return _PROVIDER_CACHE
    _PROVIDER_CACHE = {
        "binance": test_provider_connectivity("binance"),
        "kraken": test_provider_connectivity("kraken"),
    }
    _PROVIDER_CACHE_AT = time.monotonic()
    return _PROVIDER_CACHE

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
if UI_DIR.exists() and (UI_DIR / "index.html").exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


class AskModel(BaseModel):
    symbol: Optional[str] = None
    horizon: str = "24h"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (o != o or o in (float("inf"), float("-inf"))) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)[:19]
    return o


@app.get("/health")
def health():
    return {"status": "ok", "service": "neurotrom-crypto", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/crypto/status")
def status():
    live = get_live_status()
    backfill = load_backfill_state()
    
    providers = _provider_status()
    binance, kraken = providers["binance"], providers["kraken"]
    any_provider = binance.get("reachable", False) or kraken.get("reachable", False)

    # Count real and synthetic observations across the whole universe, not only
    # the first symbol. Read the provenance on persisted candles, not file age.
    df = load_universe()
    real_data_available = False
    real_rows = 0
    synthetic_rows = 0
    unknown_rows = 0
    latest_real_timestamp = None
    if not df.empty and "symbol" in df.columns:
        for symbol in df["symbol"].astype(str).tolist():
            for interval in ("1h", "1d"):
                stored = get_candles(symbol, interval, "spot")
                if stored.empty:
                    continue
                if "source_mode" not in stored.columns:
                    unknown_rows += len(stored)
                    continue
                modes = stored["source_mode"].astype(str)
                real_mask = modes.isin([SOURCE_HISTORICAL, SOURCE_LIVE])
                synth_mask = modes.eq(SOURCE_SYNTHETIC)
                real_rows += int(real_mask.sum())
                synthetic_rows += int(synth_mask.sum())
                unknown_rows += int((~real_mask & ~synth_mask).sum())
                if real_mask.any():
                    real_data_available = True
                    newest = pd.to_datetime(stored.loc[real_mask, "timestamp"], utc=True).max()
                    if pd.notna(newest) and (latest_real_timestamp is None or newest > latest_real_timestamp):
                        latest_real_timestamp = newest

    synthetic_only = synthetic_rows > 0 and real_rows == 0
    # Models
    model_provenances = []
    for h in HORIZONS:
        prov_path = MODELS / f"crypto_{h}_provenance.json"
        if prov_path.exists():
            try:
                prov = json.loads(prov_path.read_text())
                model_provenances.append(prov.get("provenance", "UNKNOWN"))
            except Exception:
                pass
    real_models = sum(1 for p in model_provenances if p in {SOURCE_LIVE, SOURCE_HISTORICAL, "REAL_MIXED"})
    
    # Predictions
    journal = compute_journal_summary()
    real_preds = journal.get("real_predictions", 0)
    
    # "Provider reachable" is not the same as "real candles ingested". Require all
    # layers to be real before ever reporting an operationally healthy system.
    live = get_live_status()
    live_updated = live.get("last_update")
    live_fresh = False
    if live_updated:
        try:
            updated_at = pd.Timestamp(live_updated)
            updated_at = updated_at.tz_localize("UTC") if updated_at.tzinfo is None else updated_at.tz_convert("UTC")
            live_fresh = (pd.Timestamp.now(tz="UTC") - updated_at).total_seconds() <= 180 and bool(live.get("streams"))
        except Exception:
            live_fresh = False

    if not real_data_available:
        overall = "BLOCKED — provider reachable but no verified real candles ingested" if any_provider else "BLOCKED — no real data provider reachable"
    elif not live_fresh:
        overall = "DEGRADED — real historical data exists, live collector is not fresh"
    elif real_models < len(HORIZONS):
        overall = "DEGRADED — real data exists, real-data models are incomplete"
    elif real_preds == 0:
        overall = "DEGRADED — no real forward predictions recorded"
    else:
        overall = "OK"

    return _clean({
        "status": "blocked" if not real_data_available else ("ok" if overall == "OK" else "degraded"),
        "overall": overall,
        "api_live": True,
        "real_data_available": real_data_available,
        "real_rows": real_rows,
        "synthetic_rows": synthetic_rows,
        "unknown_provenance_rows": unknown_rows,
        "synthetic_only": synthetic_only,
        "any_provider": any_provider,
        "providers": providers,
        "models_ready": real_models >= len(HORIZONS),
        "models_present": len(model_provenances),
        "real_models": real_models,
        "real_predictions": real_preds,
        "universe_count": len(df),
        "latest_real_candle": str(latest_real_timestamp) if latest_real_timestamp is not None else None,
        "live_collector": live,
        "backfill_last_run": backfill.get("last_run"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/v1/crypto/sources")
def sources():
    return _clean({"sources": [s.to_dict() for s in load_sources()]})


@app.get("/v1/crypto/universe")
def universe():
    df = load_universe()
    return _clean(df.to_dict(orient="records") if not df.empty else [])


@app.get("/v1/crypto/market")
def market(symbol: str = "BTCUSDT", provider: str = "auto"):
    try:
        t = fetch_ticker(symbol, "spot", provider)
        return _clean(t)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/v1/crypto/candles")
def candles(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200,
            source_mode: Optional[str] = None, allow_synthetic: bool = False):
    if source_mode == SOURCE_SYNTHETIC and not allow_synthetic:
        raise HTTPException(400, "Synthetic candles require allow_synthetic=true and are demo-only")

    if source_mode:
        df = get_candles(symbol, interval, "spot", source_mode=source_mode)
    else:
        frames = [get_candles(symbol, interval, "spot", source_mode=mode)
                  for mode in (SOURCE_HISTORICAL, SOURCE_LIVE)]
        frames = [frame for frame in frames if not frame.empty]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df.empty:
            df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    if df.empty and source_mode in (None, SOURCE_HISTORICAL):
        try:
            raw = rest_klines(symbol, interval, limit=min(max(limit, 1), 1000))
            if not raw.empty:
                save_candles(raw, symbol, interval, "spot", source_mode=SOURCE_HISTORICAL, provider="binance")
                df = get_candles(symbol, interval, "spot", source_mode=SOURCE_HISTORICAL)
        except Exception as binance_error:
            log.info("Binance candle request unavailable for %s: %s", symbol, binance_error)
        if df.empty:
            try:
                now = datetime.now(timezone.utc)
                lookback_days = 720 if interval == "1d" else 30
                backfill_symbol(symbol, interval, now - pd.Timedelta(days=lookback_days), now, "spot", provider="kraken")
                df = get_candles(symbol, interval, "spot", source_mode=SOURCE_HISTORICAL)
            except Exception as kraken_error:
                log.warning("Kraken candle request unavailable for %s: %s", symbol, kraken_error)

    if df.empty and allow_synthetic:
        df = get_candles(symbol, interval, "spot", source_mode=SOURCE_SYNTHETIC)
    if df.empty:
        raise HTTPException(503, f"No verified exchange candles available for {symbol} {interval}; synthetic data is not served by default")

    df = df.tail(max(1, min(limit, 2000)))
    fields = [col for col in ("timestamp", "open", "high", "low", "close", "volume", "source_mode", "provider") if col in df.columns]
    return _clean(df[fields].to_dict(orient="records"))


@app.get("/v1/crypto/predictions")
def predictions(symbol: Optional[str] = None, horizon: str = "24h", source_mode: Optional[str] = None):
    from journal import PREDICTIONS
    if not PREDICTIONS.exists():
        return {"predictions": [], "note": "no predictions yet"}
    df = pd.read_parquet(PREDICTIONS)
    if horizon:
        df = df[df["horizon"] == horizon]
    if symbol:
        df = df[df["symbol"] == symbol]
    if source_mode:
        df = df[df.get("source_mode", SOURCE_SYNTHETIC) == source_mode]
    else:
        # General journal views contain genuine exchange forecasts only; demo rows
        # remain accessible only through an explicit provenance filter.
        if "source_mode" not in df.columns:
            return {"predictions": [], "note": "no verifiably real predictions"}
        df = df[df["source_mode"].isin([SOURCE_LIVE, SOURCE_HISTORICAL])]
    df = df.sort_values("timestamp", ascending=False).head(200)
    return _clean({"predictions": df.to_dict(orient="records")})


@app.get("/v1/crypto/track-record")
def track_record():
    return _clean(compute_journal_summary())


@app.get("/v1/crypto/data-quality")
def data_quality():
    from universe import universe_symbols
    symbols = universe_symbols()[:20]
    results = {}
    for sym in symbols:
        for interval in ["1h", "1d"]:
            results[f"{sym}_{interval}"] = verify_backfill(sym, interval)
    return _clean(results)


@app.get("/v1/crypto/system-health")
def system_health():
    live = get_live_status()
    backfill = load_backfill_state()
    df = load_universe()
    providers = _provider_status()
    binance, kraken = providers["binance"], providers["kraken"]
    return _clean({
        "api_live": True,
        "live_collector_running": live.get("collector_running", False),
        "live_last_update": live.get("last_update"),
        "providers": {"binance": binance, "kraken": kraken},
        "backfill_last_run": backfill.get("last_run"),
        "backfill_errors": len(backfill.get("errors", [])),
        "universe_count": len(df),
        "disk_usage_mb": _disk_usage_mb(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/v1/crypto/models")
def models_status():
    result = []
    for h in HORIZONS:
        prov_path = MODELS / f"crypto_{h}_provenance.json"
        model_path = MODELS / f"crypto_{h}.txt"
        if prov_path.exists():
            try:
                prov = json.loads(prov_path.read_text())
                result.append({
                    "horizon": h,
                    "provenance": prov.get("provenance", "UNKNOWN"),
                    "trained_at": prov.get("saved_at"),
                    "model_path": prov.get("model_path"),
                })
            except Exception:
                pass
        elif model_path.exists():
            result.append({"horizon": h, "provenance": "UNKNOWN", "model_path": str(model_path)})
    return _clean(result)


@app.post("/v1/crypto/backfill")
def trigger_backfill(days: int = 7):
    try:
        from backfill import backfill_universe
        rows, errors = backfill_universe(days=days)
        return {"status": "complete", "rows": rows, "errors": len(errors)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/crypto/train")
def trigger_training(allow_synthetic: bool = False):
    try:
        report = run_training(allow_synthetic=allow_synthetic)
        return _clean(report)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/crypto/predict")
def generate_predictions(horizon: str = "24h", source_mode: Optional[str] = None):
    try:
        from features import build_feature_dataset
        df = build_feature_dataset(require_real=(source_mode != SOURCE_SYNTHETIC))
        if df.empty:
            raise HTTPException(400, "No feature data")
        
        allow_demo = source_mode == SOURCE_SYNTHETIC
        preds = predict(df, horizon, allow_synthetic=allow_demo)
        if preds is None:
            raise HTTPException(400, f"No model for horizon {horizon}")
        
        results = []
        for symbol in df["symbol"].unique():
            mask = df["symbol"] == symbol
            symbol_rows = df[mask].sort_values("timestamp")
            price_row = symbol_rows.iloc[-1] if not symbol_rows.empty else None
            if price_row is None:
                continue
            score = float(preds.loc[price_row.name])
            price = float(price_row["close"])
            src = str(price_row["source_mode"]) if "source_mode" in price_row else SOURCE_SYNTHETIC

            if not allow_demo:
                age_hours = (datetime.now(timezone.utc) - pd.Timestamp(price_row["timestamp"]).to_pydatetime()).total_seconds() / 3600
                if age_hours > 2.5:
                    continue

            record_prediction(
                symbol=symbol,
                horizon=horizon,
                timestamp=pd.Timestamp(price_row["timestamp"]).to_pydatetime(),
                predicted_return=float(score),
                model_version=f"crypto_{horizon}",
                feature_version="v1",
                price=price,
                source_mode=src,
            )
            results.append({"symbol": symbol, "score": float(score), "price": price, "source_mode": src})
        
        return _clean({"predictions": sorted(results, key=lambda x: x["score"], reverse=True)[:50], "horizon": horizon})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/crypto/live/start")
def live_start():
    start_live()
    return {"status": "started"}


@app.post("/v1/crypto/live/stop")
def live_stop():
    stop_live()
    return {"status": "stopped"}


@app.get("/v1/crypto/live/status")
def live_status():
    return _clean(get_live_status())


def _disk_usage_mb() -> float:
    try:
        total = 0
        for root, dirs, files in os.walk(str(DATA)):
            for f in files:
                fp = Path(root) / f
                if fp.exists():
                    total += fp.stat().st_size
        return round(total / (1024 * 1024), 1)
    except Exception:
        return -1


@app.get("/")
def root():
    idx = UI_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"message": "NeuroTrom Crypto API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
