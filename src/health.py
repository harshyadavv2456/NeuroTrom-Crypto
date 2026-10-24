"""Health check for NeuroTrom Crypto — exits nonzero when required conditions fail."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA, HOST, PORT, HORIZONS, SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC
from market_data import get_candles, test_provider_connectivity
from train import MODELS, _check_provenance
from journal import PREDICTIONS, SUMMARY
from live import get_live_status
from universe import load_universe

log = logging.getLogger(__name__)


def age_hours(path: Path) -> float:
    if not path.exists():
        return 1e9
    return (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "BAD"
    print(f"  {status}  {label:<40} {detail}")
    return ok


def main() -> int:
    now = datetime.now(timezone.utc)
    bad = []
    
    print(f"NeuroTrom-Crypto health  {now:%Y-%m-%d %H:%M}")
    
    # 1. API liveness
    api_ok = False
    try:
        import requests as req
        r = req.get(f"http://{HOST}:{PORT}/health", timeout=5)
        api_ok = r.status_code == 200
    except Exception:
        pass
    bad.append(not check("API liveness", api_ok, f"{HOST}:{PORT}"))
    
    # 2. Provider connectivity
    binance = test_provider_connectivity("binance")
    kraken = test_provider_connectivity("kraken")
    any_provider = binance.get("reachable", False) or kraken.get("reachable", False)
    bad.append(not check("Provider connectivity (Binance)", binance.get("reachable", False), 
                          f"latency {binance.get('latency_ms', 'N/A')}ms" if binance.get("reachable") else binance.get("error", "unreachable")))
    bad.append(not check("Provider connectivity (Kraken)", kraken.get("reachable", False),
                          f"latency {kraken.get('latency_ms', 'N/A')}ms" if kraken.get("reachable") else kraken.get("error", "unreachable")))
    
    # 3. Universe
    df = load_universe()
    bad.append(not check("Universe loaded", len(df) > 0, f"{len(df)} symbols"))
    
    # 4. Historical data coverage and provenance
    real_data_available = False
    synthetic_only = True
    if not df.empty and "symbol" in df.columns:
        sample_sym = df["symbol"].iloc[0]
        for interval in ["1h", "1d"]:
            real_df = get_candles(sample_sym, interval, "spot", source_mode=SOURCE_HISTORICAL)
            live_df = get_candles(sample_sym, interval, "spot", source_mode=SOURCE_LIVE)
            synth_df = get_candles(sample_sym, interval, "spot", source_mode=SOURCE_SYNTHETIC)
            
            has_real = not real_df.empty or not live_df.empty
            if has_real:
                real_data_available = True
                synthetic_only = False
            
            last_ts = None
            if not real_df.empty:
                last_ts = real_df["timestamp"].max().to_pydatetime()
            elif not live_df.empty:
                last_ts = live_df["timestamp"].max().to_pydatetime()
            
            age_h = (now - last_ts).total_seconds() / 3600 if last_ts else 1e9
            stale = age_h > (4 if interval == "1h" else 48)
            
            source_label = "REAL" if has_real else "SYNTHETIC"
            bad.append(not check(f"{interval} data ({source_label})", not stale, 
                                f"{len(real_df) + len(live_df) + len(synth_df)} rows, age {age_h:.1f}h, last {last_ts.isoformat()[:19] if last_ts else 'none'}"))
    
    # 5. Model readiness
    model_provenances = []
    for h in HORIZONS:
        prov_path = MODELS / f"crypto_{h}_provenance.json"
        model_path = MODELS / f"crypto_{h}.txt"
        if prov_path.exists():
            prov = json.loads(prov_path.read_text())
            model_provenances.append(prov.get("provenance", "UNKNOWN"))
        elif model_path.exists():
            model_provenances.append("UNKNOWN")
    
    models_ready = len(model_provenances) > 0
    real_models = sum(1 for p in model_provenances if p != SOURCE_SYNTHETIC)
    bad.append(not check("Models trained", models_ready, f"{len(model_provenances)} models, {real_models} real-data"))
    
    # 6. Prediction journal
    journal_ok = SUMMARY.exists()
    n_preds = 0
    n_real_preds = 0
    if PREDICTIONS.exists():
        try:
            preds = pd.read_parquet(PREDICTIONS)
            n_preds = len(preds)
            if "source_mode" in preds.columns:
                n_real_preds = len(preds[preds["source_mode"] != SOURCE_SYNTHETIC])
        except Exception:
            pass
    bad.append(not check("Prediction journal", journal_ok, f"{n_preds} total, {n_real_preds} real"))
    
    # 7. Live collector
    live = get_live_status()
    collector_running = live.get("collector_running", False)
    bad.append(not check("Live collector", collector_running, live.get("last_update", "never")))
    
    # 8. Overall system status
    if any(bad):
        status = "DEGRADED"
        if not any_provider and not real_data_available:
            status = "BLOCKED — no real data provider reachable"
    else:
        status = "OK"
    
    print("\nOverall: {status}")
    print(f"  Real data available: {'YES' if real_data_available else 'NO'}")
    print(f"  Synthetic-only mode: {'YES' if synthetic_only else 'NO'}")
    print(f"  Real predictions: {n_real_preds}")
    
    return 1 if any(bad) else 0


if __name__ == "__main__":
    sys.exit(main())
