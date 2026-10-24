"""Daily pipeline: tests -> backfill -> train -> journal -> report."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import DATA, HORIZONS
from backfill import backfill_universe
from train import run_training
from journal import mature_predictions, compute_journal_summary
from live import reconcile_gaps

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
            return {"step": name, "ok": True, "elapsed": round(elapsed, 1), "result": str(result)[:500]}
        except Exception as e:
            log.error("Step %s failed: %s", name, e)
            if attempt >= retries:
                return {"step": name, "ok": False, "error": str(e)[:500]}
            time.sleep(30)


def main():
    started = datetime.now(timezone.utc).isoformat()
    log.info("=== Daily pipeline started %s ===", started)
    
    steps = []
    
    # 1. Backfill
    steps.append(run_step("backfill", lambda: backfill_universe(days=3), retries=1))
    
    # 2. Reconcile gaps
    steps.append(run_step("reconcile", reconcile_gaps, retries=1))
    
    # 3. Train
    steps.append(run_step("train", lambda: run_training(), retries=1))
    
    # 4. Mature predictions
    for h in ["1h", "4h", "24h"]:
        steps.append(run_step(f"mature_{h}", lambda hh=h: mature_predictions(hh), retries=0))
    
    # 5. Journal summary
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
    
    log.info("=== Daily pipeline finished %s: %s ===", finished, "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())

