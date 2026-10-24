"""Decides whether the daily pipeline is due. Exit 0 = run it now, 1 = nothing to do."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_HOUR = 20


def failures_today(now):
    slot = datetime.combine(now.date(), time(RUN_HOUR)) if now.hour >= RUN_HOUR else datetime.combine(now.date() - timedelta(days=1), time(RUN_HOUR))
    n = 0
    try:
        for line in (ROOT / "models" / "run_history.jsonl").read_text().splitlines()[-30:]:
            r = json.loads(line)
            if not r.get("ok") and datetime.fromisoformat(r["started"]) >= slot:
                n += 1
    except Exception:
        pass
    return n


def code_changed_since(t):
    try:
        ts = int(subprocess.run(["git", "log", "-1", "--format=%ct"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
        return datetime.fromtimestamp(ts) > t
    except Exception:
        return False


def due(now=None):
    now = now or datetime.now()
    if subprocess.run(["pgrep", "-f", r"python src/daily\.py$"], capture_output=True).stdout.strip():
        return False, "already running"
    try:
        s = json.loads((ROOT / "models/daily_status.json").read_text())
        started = datetime.fromisoformat(s["started"])
        ok = bool(s.get("ok")) and s.get("finished")
    except Exception:
        return True, "no previous status"
    
    if s.get("publish_pending") and s.get("artifacts_ok"):
        return False, "publish pending"
    
    failed = failures_today(now)
    if not ok and failed < 4:
        return False, f"recent failure ({failed} today)"
    
    return True, "due"


def main():
    ok, reason = due()
    if ok:
        print(f"Daily pipeline is due: {reason}")
        sys.exit(0)
    else:
        print(f"Daily pipeline not due: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
