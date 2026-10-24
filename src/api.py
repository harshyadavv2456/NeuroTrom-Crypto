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
    
    # Provider connectivity (cached to avoid slow status page)
    from functools import lru_cache
    @lru_cache(maxsize=1)
    def _cached_providers():
        return {"binance": test_provider_connectivity("binance"), "kraken": test_provider_connectivity("kraken")}
    providers = _cached_providers()
    binance, kraken = providers["binance"], providers["kraken"]
    any_provider = binance.get("reachable", False) or kraken.get("reachable", False)
    
    # Data provenance
    df = load_universe()
    real_data_available = False
    synthetic_only = True
    if not df.empty and "symbol" in df.columns:
        sample_sym = df["symbol"].iloc[0]
        for interval in ["1h", "1d"]:
            real_df = get_candles(sample_sym, interval, "spot", source_mode=SOURCE_HISTORICAL)
            live_df = get_candles(sample_sym, interval, "spot", source_mode=SOURCE_LIVE)
            if not real_df.empty or not live_df.empty:
                real_data_available = True
                synthetic_only = False
    
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
    real_models = sum(1 for p in model_provenances if p != SOURCE_SYNTHETIC)
    
    # Predictions
    journal = compute_journal_summary()
    real_preds = journal.get("real_predictions", 0)
    
    # Overall
    if not any_provider and not real_data_available:
        overall = "BLOCKED — no real data provider reachable"
    elif real_data_available and real_models > 0 and real_preds > 0:
        overall = "OK"
    elif real_data_available:
        overall = "DEGRADED — real data available, no real predictions yet"
    else:
        overall = "DEGRADED — synthetic data only"
    
    return _clean({
        "status": "ok",
        "overall": overall,
        "api_live": True,
        "real_data_available": real_data_available,
        "synthetic_only": synthetic_only,
        "any_provider": any_provider,
        "providers": {"binance": binance, "kraken": kraken},
        "models_ready": len(model_provenances) > 0,
        "real_models": real_models,
        "real_predictions": real_preds,
        "universe_count": len(df),
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
def market(symbol: str = "BTCUSDT", provider: str = "binance"):
    try:
        t = fetch_ticker(symbol, "spot", provider)
        return _clean(t)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/v1/crypto/candles")
def candles(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200, source_mode: Optional[str] = None):
    df = get_candles(symbol, interval, "spot", source_mode=source_mode)
    if df.empty:
        try:
            raw = rest_klines(symbol, interval, limit=limit)
            if not raw.empty:
                save_candles(raw, symbol, interval, "spot", source_mode=SOURCE_HISTORICAL, provider="binance")
                df = raw
        except Exception:
            pass
    if df.empty:
        raise HTTPException(404, f"No candles for {symbol} {interval}")
    df = df.tail(limit)
    return _clean(df[["timestamp", "open", "high", "low", "close", "volume", "source_mode", "provider"]].to_dict(orient="records"))


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
    providers = _cached_providers()
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
        
        preds = predict(df, horizon)
        if preds is None:
            raise HTTPException(400, f"No model for horizon {horizon}")
        
        results = []
        for symbol in df["symbol"].unique():
            mask = df["symbol"] == symbol
            score = preds[mask].mean()
            price_row = df[mask].iloc[-1] if not df[mask].empty else None
            price = float(price_row["close"]) if price_row is not None else 0.0
            src = df[mask]["source_mode"].iloc[0] if "source_mode" in df.columns and not df[mask].empty else SOURCE_SYNTHETIC
            
            record_prediction(
                symbol=symbol,
                horizon=horizon,
                timestamp=datetime.now(timezone.utc),
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
