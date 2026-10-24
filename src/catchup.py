"""Due-check for the isolated Crypto daily job.

Uses the same state path as daily.py, compares timezone-aware timestamps, and
runs once per daily slot with bounded retries after failures.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, time, timedelta
from pathlib import Path

from config import DATA

RUN_HOUR = 20
STATUS_PATH = DATA / "models" / "daily_status.json"
RUN_HISTORY = DATA / "models" / "run_history.jsonl"


def _aware(value: str, tzinfo) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=tzinfo) if parsed.tzinfo is None else parsed.astimezone(tzinfo)


def failures_today(now: datetime) -> int:
    """Count failed runs since the current scheduled slot began."""
    slot_day = now.date() if now.hour >= RUN_HOUR else (now - timedelta(days=1)).date()
    slot = datetime.combine(slot_day, time(RUN_HOUR), tzinfo=now.tzinfo)
    failures = 0
    try:
        for line in RUN_HISTORY.read_text().splitlines()[-100:]:
            try:
                run = json.loads(line)
                started = _aware(run["started"], now.tzinfo)
                if not run.get("ok") and started >= slot:
                    failures += 1
            except (ValueError, KeyError, TypeError):
                continue
    except OSError:
        pass
    return failures


def due(now: datetime | None = None) -> tuple[bool, str]:
    now = now or datetime.now().astimezone()

    active = subprocess.run(
        ["pgrep", "-f", r"[p]ython.*src/daily\.py"],
        capture_output=True, text=True,
    ).stdout.strip()
    if active:
        return False, "daily pipeline already running"

    if not STATUS_PATH.exists():
        return True, "no previous daily status"

    try:
        status = json.loads(STATUS_PATH.read_text())
        started = _aware(status["started"], now.tzinfo)
        finished_raw = status.get("finished")
        finished = _aware(finished_raw, now.tzinfo) if finished_raw else started
    except (OSError, ValueError, KeyError, TypeError):
        return True, "no valid previous daily status"

    slot_day = now.date() if now.hour >= RUN_HOUR else (now - timedelta(days=1)).date()
    slot = datetime.combine(slot_day, time(RUN_HOUR), tzinfo=now.tzinfo)

    if started >= slot and bool(status.get("ok")) and finished:
        return False, "daily slot already completed"

    if started >= slot and not status.get("ok"):
        failures = failures_today(now)
        age_hours = max(0.0, (now - finished).total_seconds() / 3600)
        if failures >= 4:
            return False, f"retry limit reached ({failures} failures)"
        if age_hours < 3:
            return False, f"failed recently; retry after 3 hours ({age_hours:.1f}h elapsed)"
        return True, f"retrying failed daily run ({failures} failures)"

    if now < datetime.combine(now.date(), time(RUN_HOUR), tzinfo=now.tzinfo):
        return False, "waiting for the 20:00 local daily slot"
    return True, "daily slot is due"


def main() -> None:
    import sys
    should_run, reason = due()
    print(f"{'RUN' if should_run else 'SKIP'}: {reason}")
    sys.exit(0 if should_run else 1)


if __name__ == "__main__":
    main()
