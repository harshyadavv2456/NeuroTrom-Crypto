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

from config import DATA, HOST, PORT, HORIZONS
from market_data import get_candles, rest_klines, save_candles, fetch_ticker
from universe import load_universe, build_universe
from backfill import backfill_symbol, verify_backfill, backfill_universe, load_backfill_state
from features import build_feature_dataset, load_features
from train import run_training, predict, load_features as _lf
from journal import compute_journal_summary, record_prediction, mature_predictions
from live import get_live_status, start_live, stop_live, reconcile_gaps

log = logging.getLogger(__name__)
app = FastAPI(title="NeuroTrom Crypto", version="0.1.0")

# Serve dashboard
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
    return _clean({
        "status": "ok",
        "live": live,
        "backfill_last_run": backfill.get("last_run"),
        "universe_count": len(load_universe()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/v1/crypto/universe")
def universe():
    df = load_universe()
    return _clean(df.to_dict(orient="records") if not df.empty else [])


@app.get("/v1/crypto/market")
def market(symbol: str = "BTCUSDT"):
    try:
        t = fetch_ticker(symbol, "spot")
        return _clean(t)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/v1/crypto/candles")
def candles(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200):
    df = get_candles(symbol, interval, "spot")
    if df.empty:
        # Try live fetch
        try:
            raw = rest_klines(symbol, interval, limit=limit)
            if not raw.empty:
                save_candles(raw, symbol, interval, "spot")
                df = raw
        except Exception:
            pass
    if df.empty:
        raise HTTPException(404, f"No candles for {symbol} {interval}")
    df = df.tail(limit)
    return _clean(df[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="records"))


@app.get("/v1/crypto/predictions")
def predictions(symbol: Optional[str] = None, horizon: str = "24h"):
    from journal import PREDICTIONS
    if not PREDICTIONS.exists():
        return {"predictions": [], "note": "no predictions yet"}
    df = pd.read_parquet(PREDICTIONS)
    if horizon:
        df = df[df["horizon"] == horizon]
    if symbol:
        df = df[df["symbol"] == symbol]
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
    return _clean({
        "live_collector_running": live.get("collector_running", False),
        "live_last_update": live.get("last_update"),
        "backfill_last_run": backfill.get("last_run"),
        "backfill_errors": len(backfill.get("errors", [])),
        "universe_count": len(df),
        "disk_usage_mb": _disk_usage_mb(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.post("/v1/crypto/backfill")
def trigger_backfill(days: int = 7):
    """Trigger a backfill job."""
    try:
        from backfill import backfill_universe
        rows, errors = backfill_universe(days=days)
        return {"status": "complete", "rows": rows, "errors": len(errors)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/crypto/train")
def trigger_training():
    """Trigger model training."""
    try:
        report = run_training()
        return _clean(report)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/crypto/predict")
def generate_predictions(horizon: str = "24h"):
    """Generate predictions for all symbols in universe."""
    try:
        df = build_feature_dataset()
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
            
            record_prediction(
                symbol=symbol,
                horizon=horizon,
                timestamp=datetime.now(timezone.utc),
                predicted_return=float(score),
                model_version=f"crypto_{horizon}",
                feature_version="v1",
                price=price,
            )
            results.append({"symbol": symbol, "score": float(score), "price": price})
        
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
        for root, dirs, files in os_walk(str(DATA)):
            for f in files:
                fp = Path(root) / f
                if fp.exists():
                    total += fp.stat().st_size
        return round(total / (1024 * 1024), 1)
    except Exception:
        return -1


def os_walk(path):
    import os
    return os.walk(path)


# Serve index.html at root
@app.get("/")
def root():
    idx = UI_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"message": "NeuroTrom Crypto API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

