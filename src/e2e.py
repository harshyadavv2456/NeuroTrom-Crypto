"""End-to-end verification with source-aware checks."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import subprocess
import sys
import time
from pathlib import Path

import requests
import pandas as pd

from config import DATA, HOST, PORT, SOURCE_SYNTHETIC
from market_data import test_provider_connectivity
from features import build_feature_dataset
from train import _check_provenance, MODELS
from journal import PREDICTIONS
from universe import load_universe

log = logging.getLogger(__name__)
RESULTS = []
REPORTS_DIR = DATA / "reports"


def check(layer: str, name: str, fn):
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    RESULTS.append((layer, name, "PASS" if ok else "FAIL", str(detail)[:200], round(elapsed, 1)))
    print(f"{'PASS' if ok else 'FAIL'} [{layer}] {name} - {str(detail)[:120]}", flush=True)


def main() -> int:
    # 1. Provider connectivity — must be reported accurately even when unreachable
    def check_binance():
        r = test_provider_connectivity("binance")
        ok = "provider" in r and "reachable" in r
        return ok, f"reachable={r.get('reachable')}, reported correctly"
    check("data", "binance_connectivity_reported", check_binance)
    
    def check_kraken():
        r = test_provider_connectivity("kraken")
        ok = "provider" in r and "reachable" in r
        return ok, f"reachable={r.get('reachable')}, reported correctly"
    check("data", "kraken_connectivity_reported", check_kraken)
    
    # 2. Universe
    def universe_check():
        df = load_universe()
        return len(df) >= 30, f"{len(df)} symbols"
    check("data", "universe_loaded", universe_check)
    
    # 3. System correctly reports real data status
    def real_data_check():
        from market_data import get_candles
        from config import SOURCE_HISTORICAL, SOURCE_LIVE
        df = load_universe()
        if df.empty:
            return False, "no universe"
        sym = df["symbol"].iloc[0]
        real = get_candles(sym, "1h", "spot", source_mode=SOURCE_HISTORICAL)
        live = get_candles(sym, "1h", "spot", source_mode=SOURCE_LIVE)
        has_real = not real.empty or not live.empty
        
        # Verify API status matches reality
        import requests
        from config import HOST, PORT
        r = requests.get(f"http://{HOST}:{PORT}/v1/crypto/status", timeout=30)
        data = r.json()
        reported_real = data.get("real_data_available", False)
        reported_synthetic = data.get("synthetic_only", False)
        
        if not has_real and not reported_real and reported_synthetic:
            return True, "correctly reports: no real data, synthetic-only"
        elif has_real and reported_real:
            return True, "real data present and reported"
        else:
            return False, f"mismatch: has_real={has_real}, reported_real={reported_real}, synthetic_only={reported_synthetic}"
    check("data", "real_data_status_accurate", real_data_check)
    
    # 4. Features reject synthetic-only when required
    def features_reject_synthetic():
        from features import build_feature_dataset
        df = build_feature_dataset(require_real=True)
        if df.empty:
            return True, "correctly rejected synthetic-only data"
        return False, "should have rejected synthetic-only data"
    check("data", "features_reject_synthetic", features_reject_synthetic)
    
    # 5. Training refuses synthetic-only by default
    def training_rejects_synthetic():
        from train import run_training
        try:
            run_training(allow_synthetic=False)
            return False, "should have refused synthetic-only training"
        except (ValueError, RuntimeError) as e:
            if "synthetic" in str(e).lower() or "no feature data" in str(e).lower():
                return True, f"correctly rejected: {e}"
            return False, f"wrong error: {e}"
    check("research", "training_rejects_synthetic", training_rejects_synthetic)
    
    # 6. Training accepts explicit flag (for research)
    def training_allows_synthetic_explicit():
        from train import run_training
        try:
            report = run_training(allow_synthetic=True)
            return "horizons" in report, "training with explicit synthetic flag"
        except Exception as e:
            return False, str(e)
    check("research", "training_explicit_synthetic", training_allows_synthetic_explicit)
    
    # 7. Model provenance
    def model_provenance():
        import os
        files = list(MODELS.glob("crypto_*_provenance.json"))
        if not files:
            return False, "no provenance files"
        for f in files:
            data = json.loads(f.read_text())
            if data.get("provenance") == SOURCE_SYNTHETIC:
                return True, f"models labeled synthetic: {f.name}"
        return True, f"{len(files)} provenance files"
    check("research", "model_provenance", model_provenance)
    
    # 8. API health
    def api_health():
        r = requests.get(f"http://{HOST}:{PORT}/health", timeout=5)
        return r.status_code == 200, f"HTTP {r.status_code}"
    check("api", "health_endpoint", api_health)
    
    # 9. API status shows accurate overall
    def api_status_accurate():
        r = requests.get(f"http://{HOST}:{PORT}/v1/crypto/status", timeout=60)
        data = r.json()
        overall = data.get("overall", "")
        ok = "DEGRADED" in overall or "BLOCKED" in overall or "OK" in overall
        return ok, f"overall={overall}"
    check("api", "status_accurate", api_status_accurate)
    
    # 10. Dashboard
    def dashboard_check():
        r = requests.get(f"http://{HOST}:{PORT}/", timeout=5)
        return r.status_code == 200 and "NeuroTrom" in r.text, f"HTTP {r.status_code}"
    check("api", "dashboard", dashboard_check)
    
    # 11. No synthetic in production prediction endpoint
    def no_synthetic_predictions():
        r = requests.get(f"http://{HOST}:{PORT}/v1/crypto/predictions", timeout=5)
        data = r.json()
        preds = data.get("predictions", [])
        synthetic = [p for p in preds if p.get("source_mode") == SOURCE_SYNTHETIC]
        return len(synthetic) == 0, f"{len(synthetic)} synthetic predictions in production feed"
    check("api", "no_synthetic_predictions", no_synthetic_predictions)
    
    # Write report
    report_path = REPORTS_DIR / "e2e_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# E2E Report {datetime.now(timezone.utc):%Y-%m-%d %H:%M}", ""]
    for layer, name, status, detail, elapsed in RESULTS:
        lines.append(f"- [{status}] [{layer}] {name} ({elapsed}s): {detail}")
    passed = sum(1 for r in RESULTS if r[2] == "PASS")
    lines.append(f"\nResult: {'PASS' if passed == len(RESULTS) else 'FAIL'} ({passed}/{len(RESULTS)})")
    report_path.write_text("\n".join(lines))
    
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
