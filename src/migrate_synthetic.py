"""One-time, auditable migration of legacy synthetic/unprovenanced candles.

Production paths are rewritten to contain exchange-provenanced rows only.
Synthetic rows are copied and verified in the isolated demo archive. Rows with no
trusted provenance are preserved in a quarantine tree, never relabelled as real.
The operation is idempotent and can be rerun safely.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import DATA, SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC
from market_data import RAW, MARKET, SYNTHETIC

log = logging.getLogger(__name__)
REPORT_PATH = DATA / "reports" / "synthetic_migration.json"
REAL_MODES = {SOURCE_LIVE, SOURCE_HISTORICAL}


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, engine="pyarrow")
    tmp.replace(path)


def _archive_rows(frame: pd.DataFrame, dest: Path, time_col: str) -> int:
    if frame.empty:
        return 0
    piece = frame.copy()
    if time_col in piece.columns:
        piece[time_col] = pd.to_datetime(piece[time_col], utc=True, errors="coerce")
    if dest.exists():
        old = pd.read_parquet(dest)
        if time_col in old.columns:
            old[time_col] = pd.to_datetime(old[time_col], utc=True, errors="coerce")
        piece = pd.concat([old, piece], ignore_index=True)
    if time_col in piece.columns:
        piece = piece.drop_duplicates(subset=[time_col], keep="last").sort_values(time_col)
    _atomic_parquet(piece, dest)
    # Verify the archive persisted the unique candle timestamps before production
    # is rewritten. Never discard source rows if the archive could not be verified.
    check = pd.read_parquet(dest)
    if time_col in check.columns:
        expected = set(pd.to_datetime(piece[time_col], utc=True, errors="coerce").dropna())
        actual = set(pd.to_datetime(check[time_col], utc=True, errors="coerce").dropna())
        if not expected.issubset(actual):
            raise RuntimeError(f"Archive verification failed for {dest}")
    return len(frame)


def _migrate_one(path: Path, root: Path, archive_root: Path, quarantine_root: Path,
                 time_col: str, report: dict) -> None:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        report["errors"].append({"path": str(path), "error": f"read: {exc}"})
        return
    if frame.empty:
        path.unlink(missing_ok=True)
        report["empty_files_removed"] += 1
        return

    relative = path.relative_to(root)
    mode = frame["source_mode"].astype(str) if "source_mode" in frame.columns else pd.Series(["UNKNOWN"] * len(frame), index=frame.index)
    real_mask = mode.isin(REAL_MODES)
    demo_mask = mode.eq(SOURCE_SYNTHETIC)
    unknown_mask = ~(real_mask | demo_mask)

    real = frame.loc[real_mask].copy()
    demo = frame.loc[demo_mask].copy()
    unknown = frame.loc[unknown_mask].copy()

    if not demo.empty:
        destination = archive_root / relative
        try:
            _archive_rows(demo, destination, time_col)
            report["synthetic_rows_archived"] += len(demo)
        except Exception as exc:
            report["errors"].append({"path": str(path), "error": f"archive synthetic: {exc}"})
            return

    if not unknown.empty:
        destination = quarantine_root / relative
        try:
            _archive_rows(unknown, destination, time_col)
            report["unknown_rows_quarantined"] += len(unknown)
        except Exception as exc:
            report["errors"].append({"path": str(path), "error": f"quarantine unknown: {exc}"})
            return

    if len(real) != len(frame):
        if real.empty:
            path.unlink(missing_ok=True)
        else:
            _atomic_parquet(real, path)
        report["production_files_rewritten"] += 1

    report["source_rows_seen"] += len(frame)
    report["real_rows_retained"] += len(real)
    if len(real) != len(frame):
        report["files_with_nonreal_rows"] += 1


def _active_store_audit() -> dict:
    result = {
        "files_checked": 0,
        "active_real_rows": 0,
        "active_synthetic_rows": 0,
        "active_unknown_rows": 0,
        "examples": [],
    }
    for root in (RAW, MARKET):
        if not root.exists():
            continue
        for path in root.rglob("*.parquet"):
            if path.name.endswith(".tmp") or ".lock" in path.name:
                continue
            result["files_checked"] += 1
            try:
                frame = pd.read_parquet(path)
                if "source_mode" not in frame.columns:
                    result["active_unknown_rows"] += len(frame)
                    if len(result["examples"]) < 20:
                        result["examples"].append({"path": str(path), "status": "missing_provenance", "rows": len(frame)})
                    continue
                modes = frame["source_mode"].astype(str)
                real = modes.isin(REAL_MODES)
                synthetic = modes.eq(SOURCE_SYNTHETIC)
                unknown = ~(real | synthetic)
                result["active_real_rows"] += int(real.sum())
                result["active_synthetic_rows"] += int(synthetic.sum())
                result["active_unknown_rows"] += int(unknown.sum())
                if bool((synthetic | unknown).any()) and len(result["examples"]) < 20:
                    result["examples"].append({
                        "path": str(path),
                        "synthetic_rows": int(synthetic.sum()),
                        "unknown_rows": int(unknown.sum()),
                    })
            except Exception as exc:
                result["active_unknown_rows"] += 1
                if len(result["examples"]) < 20:
                    result["examples"].append({"path": str(path), "status": "read_error", "error": str(exc)[:200]})
    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_rows_seen": 0,
        "real_rows_retained": 0,
        "synthetic_rows_archived": 0,
        "unknown_rows_quarantined": 0,
        "production_files_rewritten": 0,
        "empty_files_removed": 0,
        "files_with_nonreal_rows": 0,
        "errors": [],
    }
    quarantine_raw = DATA / "quarantine" / "raw" / "market"
    quarantine_market = DATA / "quarantine" / "market"

    # The two production trees have different relative roots and timestamp columns.
    for root, archive_root, quarantine_root, time_col in (
        (RAW, SYNTHETIC / "raw" / "market", quarantine_raw, "open_time"),
        (MARKET, SYNTHETIC / "market", quarantine_market, "timestamp"),
    ):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.parquet")):
            if path.name.endswith(".tmp") or ".lock" in path.name:
                continue
            _migrate_one(path, root, archive_root, quarantine_root, time_col, report)

    report["production_store_audit"] = _active_store_audit()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "ok" if (
        not report["errors"]
        and report["production_store_audit"]["active_synthetic_rows"] == 0
        and report["production_store_audit"]["active_unknown_rows"] == 0
    ) else "failed"

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str))
    tmp.replace(REPORT_PATH)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
