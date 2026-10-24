"""Health check for NeuroTrom Crypto."""
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

from config import DATA, HOST, PORT, HORIZONS
from market_data import get_candles
from train import MODELS
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
    print(f"  {status}  {label:<36} {detail}")
    return ok


def main() -> int:
    now = datetime.now(timezone.utc)
    bad = []
    
    print(f"NeuroTrom-Crypto health  {now:%Y-%m-%d %H:%M}")
    
    # Service listening
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect((HOST, PORT))
        ok = True
    except Exception:
        ok = False
    bad.append(not check("API service", ok, f"{HOST}:{PORT}"))
    
    # Universe
    df = load_universe()
    ok = len(df) > 0
    bad.append(not check("universe", ok, f"{len(df)} symbols"))
    
    # Models
    model_ok = all((MODELS / f"crypto_{h}.txt").exists() for h in HORIZONS if h != "7d")
    bad.append(not check("models", model_ok, f"{sum(1 for h in HORIZONS if (MODELS / f'crypto_{h}.txt').exists())}/{len(HORIZONS)}"))
    
    # Data freshness
    if not df.empty and "symbol" in df.columns:
        sym = df["symbol"].iloc[0]
        last_1h = None
        last_1d = None
        try:
            c = get_candles(sym, "1h", "spot")
            if not c.empty:
                last_1h = c["timestamp"].max().to_pydatetime()
        except Exception:
            pass
        try:
            c = get_candles(sym, "1d", "spot")
            if not c.empty:
                last_1d = c["timestamp"].max().to_pydatetime()
        except Exception:
            pass
        
        age_1h = (now - last_1h).total_seconds() / 3600 if last_1h else 1e9
        age_1d = (now - last_1d).total_seconds() / 3600 if last_1d else 1e9
        bad.append(not check("1h data fresh", age_1h < 4, f"{age_1h:.1f}h old"))
        bad.append(not check("1d data fresh", age_1d < 48, f"{age_1d:.1f}h old"))
    
    # Predictions
    ok = PREDICTIONS.exists()
    n_preds = 0
    if ok:
        try:
            n_preds = len(pd.read_parquet(PREDICTIONS))
        except Exception:
            pass
    bad.append(not check("predictions", ok, f"{n_preds} records"))
    
    # Journal summary
    ok = SUMMARY.exists()
    bad.append(not check("journal summary", ok))
    
    # Live collector
    live = get_live_status()
    bad.append(not check("live collector", live.get("collector_running", False), live.get("last_update", "never")))
    
    # Disk
    try:
        total = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
        mb = total / (1024 * 1024)
        ok = mb < 50 * 1024  # 50 GB limit
        bad.append(not check("disk usage", ok, f"{mb:.0f} MB"))
    except Exception:
        pass
    
    if any(bad):
        print(f"\nNeeds attention: {sum(bad)} checks failed")
        return 1
    print("\nAll good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

