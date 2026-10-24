"""Daily pipeline: tests -> source check -> backfill -> validate -> features -> train -> journal -> report."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import DATA, HORIZONS
from market_data import test_provider_connectivity
from backfill import backfill_universe
from features import build_feature_dataset
from train import run_training
from journal import mature_predictions, compute_journal_summary
from live import reconcile_gaps
from train import MODELS
from config import SOURCE_LIVE, SOURCE_HISTORICAL
from universe import load_universe

log = logging.getLogger(__name__)
MODELS = DATA / "models"
STATUS = MODELS / "daily_status.json"
REPORTS = DATA / "reports"


def run_step(name: str, fn, retries: int = 1):
    for attempt in range(retries + 1):
        try:
            log.info("Step %s (attempt %d)...", name, attempt + 1)
            t0 = time.time()
            result = fn()
            elapsed = time.time() - t0
            log.info("Step %s OK in %.1fs", name, elapsed)
            safe_result = result if isinstance(result, (dict, list, int, float, bool)) or result is None else str(result)[:500]
            return {"step": name, "ok": True, "elapsed": round(elapsed, 1), "result": safe_result}
        except Exception as e:
            log.error("Step %s failed: %s", name, e)
            if attempt >= retries:
                return {"step": name, "ok": False, "error": str(e)[:500]}
            time.sleep(30)


def _all_real_models_ready() -> bool:
    real_modes = {SOURCE_LIVE, SOURCE_HISTORICAL, "REAL_MIXED"}
    for horizon in HORIZONS:
        model_file = MODELS / f"crypto_{horizon}.txt"
        provenance_file = MODELS / f"crypto_{horizon}_provenance.json"
        if not model_file.is_file() or not provenance_file.is_file():
            return False
        try:
            metadata = json.loads(provenance_file.read_text())
        except (OSError, ValueError):
            return False
        if metadata.get("provenance") not in real_modes:
            return False
    return True


def main():
    started = datetime.now(timezone.utc).isoformat()
    log.info("=== Daily pipeline started %s ===", started)

    steps = []

    # First-run/recovery bootstrap is only invoked while one or more real-data
    # models are missing. It fails honestly and quickly if no provider is reachable.
    if not _all_real_models_ready():
        def bootstrap_real():
            from bootstrap_real import main as bootstrap_main
            exit_code = bootstrap_main()
            if exit_code != 0:
                raise RuntimeError(f"real-data bootstrap returned exit code {exit_code}")
            return {"status": "complete"}
        steps.append(run_step("real_bootstrap_if_needed", bootstrap_real, retries=0))
    
    # 1. Source connectivity
    steps.append(run_step("source_check", lambda: {
        "binance": test_provider_connectivity("binance"),
        "kraken": test_provider_connectivity("kraken"),
    }, retries=0))
    
    # 2. Backfill (only if provider reachable)
    connectivity = steps[-1].get("result", {})
    try:
        conn = json.loads(connectivity) if isinstance(connectivity, str) else connectivity
    except Exception:
        conn = {}
    
    any_provider = any(v.get("reachable") for v in conn.values() if isinstance(v, dict))
    if any_provider:
        steps.append(run_step("backfill", lambda: backfill_universe(days=3), retries=1))
        steps.append(run_step("reconcile", reconcile_gaps, retries=1))
    else:
        steps.append({"step": "backfill", "ok": False, "error": "no provider reachable"})
        steps.append({"step": "reconcile", "ok": False, "error": "no provider reachable"})
    
    # 3. Features (requires real data)
    def build_features():
        df = build_feature_dataset(require_real=True)
        if df.empty:
            raise RuntimeError("No real feature data available")
        return f"{len(df)} rows"
    steps.append(run_step("features", build_features, retries=0))
    
    # 4. Train (only if real data)
    features_ok = steps[-1].get("ok", False)
    if features_ok:
        steps.append(run_step("train", lambda: run_training(allow_synthetic=False), retries=1))
    else:
        steps.append({"step": "train", "ok": False, "error": "skipped: no real features"})
    
    # 5. Mature predictions
    for h in ["1h", "4h", "24h", "7d"]:
        steps.append(run_step(f"mature_{h}", lambda hh=h: mature_predictions(hh), retries=0))
    
    # 6. Journal summary
    steps.append(run_step("journal", compute_journal_summary, retries=0))
    
    ok = all(s.get("ok") for s in steps)
    finished = datetime.now(timezone.utc).isoformat()
    
    status = {
        "started": started,
        "finished": finished,
        "ok": ok,
        "steps": {s["step"]: s for s in steps},
    }
    STATUS.write_text(json.dumps(status, indent=2, default=str))
    history_path = DATA / "models" / "run_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as history:
        history.write(json.dumps({"started": started, "finished": finished, "ok": ok,
                                  "steps": {step["step"]: step.get("ok", False) for step in steps}},
                                 default=str) + "\n")
    
    log.info("=== Daily pipeline finished %s: %s ===", finished, "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
