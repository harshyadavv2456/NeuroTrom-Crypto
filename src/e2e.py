"""End-to-end verification."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import DATA, HOST, PORT

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
    # Data source smoke test
    def ds_smoke():
        from market_data import rest_klines
        df = rest_klines("BTCUSDT", "1h", limit=5)
        return len(df) >= 5, f"fetched {len(df)} candles"
    check("data", "binance_rest_smoke", ds_smoke)
    
    # Universe
    def universe_check():
        from universe import build_universe
        df = build_universe(30)
        return len(df) > 0, f"{len(df)} symbols"
    check("data", "universe_discovery", universe_check)
    
    # Backfill
    def backfill_check():
        from backfill import backfill_universe
        rows, errors = backfill_universe(days=1)
        return rows > 0, f"{rows} rows, {len(errors)} errors"
    check("data", "backfill", backfill_check)
    
    # Features
    def features_check():
        from features import build_feature_dataset, save_features
        df = build_feature_dataset(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        return len(df) > 0, f"{len(df)} rows, {len(df.columns)} cols"
    check("research", "features", features_check)
    
    # Training
    def training_check():
        from train import run_training
        report = run_training()
        return "horizons" in report and any("error" not in v for v in report["horizons"].values()), "models trained"
    check("research", "training", training_check)
    
    # Journal
    def journal_check():
        from journal import compute_journal_summary
        s = compute_journal_summary()
        return "total_predictions" in s, f"{s.get('total_predictions', 0)} predictions"
    check("research", "journal", journal_check)
    
    # API (if running)
    def api_check():
        r = requests.get(f"http://{HOST}:{PORT}/health", timeout=5)
        return r.status_code == 200, f"HTTP {r.status_code}"
    check("api", "health_endpoint", api_check)
    
    # Dashboard
    def dashboard_check():
        r = requests.get(f"http://{HOST}:{PORT}/", timeout=5)
        return r.status_code == 200 and "NeuroTrom" in r.text, f"HTTP {r.status_code}"
    check("api", "dashboard", dashboard_check)
    
    # Write report
    report_path = REPORTS_DIR / "e2e_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# E2E Report {datetime.now(timezone.utc):%Y-%m-%d %H:%M}", ""]
    for layer, name, status, detail, elapsed in RESULTS:
        lines.append(f"- [{status}] [{layer}] {name} ({elapsed}s): {detail}")
    lines.append(f"\nResult: {'PASS' if all(r[2] == 'PASS' for r in RESULTS) else 'FAIL'} ({sum(1 for r in RESULTS if r[2] == 'PASS')}/{len(RESULTS)})")
    report_path.write_text("\n".join(lines))
    
    return 0 if all(r[2] == "PASS" for r in RESULTS) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
