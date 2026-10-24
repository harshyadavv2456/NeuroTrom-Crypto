"""Generate idempotent forward crypto forecasts from fresh real candles only."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from config import DATA, HORIZONS, SOURCE_LIVE, SOURCE_HISTORICAL
from features import build_feature_dataset
from journal import record_prediction
from train import MODELS, predict

log = logging.getLogger(__name__)
REAL_MODES = {SOURCE_LIVE, SOURCE_HISTORICAL, "REAL_MIXED"}
MAX_INPUT_AGE_HOURS = 2.5


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    now = datetime.now(timezone.utc)
    report = {
        "started_at": now.isoformat(),
        "status": "blocked",
        "created_or_already_recorded": 0,
        "skipped": {},
        "by_horizon": {},
    }

    features = build_feature_dataset(interval="1h", require_real=True)
    if features.empty or "source_mode" not in features.columns:
        report["reason"] = "No exchange-only features available"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(report)
        print(json.dumps(report, indent=2))
        return 2

    features = features.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True, errors="coerce")
    features = features.dropna(subset=["timestamp"])
    source_modes = set(features["source_mode"].astype(str).unique())
    if not source_modes or not source_modes.issubset({SOURCE_LIVE, SOURCE_HISTORICAL}):
        report["reason"] = f"Feature provenance is not real-only: {sorted(source_modes)}"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(report)
        print(json.dumps(report, indent=2))
        return 3

    provenance_report = {}
    created = 0
    for horizon in HORIZONS:
        model_path = MODELS / f"crypto_{horizon}.txt"
        provenance_path = MODELS / f"crypto_{horizon}_provenance.json"
        try:
            meta = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
        except (OSError, ValueError):
            meta = {}
        model_provenance = str(meta.get("provenance", "UNKNOWN"))
        provenance_report[horizon] = {
            "model_file_exists": model_path.is_file(),
            "provenance": model_provenance,
            "trained_at": meta.get("saved_at"),
        }
        if not model_path.is_file() or model_provenance not in REAL_MODES:
            report["skipped"][horizon] = f"Missing model or non-real model provenance ({model_provenance})"
            report["by_horizon"][horizon] = {"recorded": 0}
            continue

        try:
            scores = predict(features, horizon, allow_synthetic=False)
            if scores is None:
                report["skipped"][horizon] = "No compatible prediction model"
                report["by_horizon"][horizon] = {"recorded": 0}
                continue
        except Exception as exc:
            report["skipped"][horizon] = f"{type(exc).__name__}: {str(exc)[:240]}"
            report["by_horizon"][horizon] = {"recorded": 0}
            continue

        horizon_created = 0
        horizon_skipped = 0
        for symbol in features["symbol"].astype(str).unique():
            symbol_rows = features[features["symbol"].astype(str) == symbol].sort_values("timestamp")
            if symbol_rows.empty:
                continue
            row = symbol_rows.iloc[-1]
            cutoff = pd.Timestamp(row["timestamp"])
            age_hours = max(0.0, (pd.Timestamp.now(tz="UTC") - cutoff).total_seconds() / 3600)
            mode = str(row["source_mode"])
            if mode not in {SOURCE_LIVE, SOURCE_HISTORICAL} or age_hours > MAX_INPUT_AGE_HOURS:
                horizon_skipped += 1
                continue
            if row["close"] is None or not pd.notna(row["close"]) or float(row["close"]) <= 0:
                horizon_skipped += 1
                continue
            try:
                score = float(scores.loc[row.name])
            except (KeyError, TypeError, ValueError):
                horizon_skipped += 1
                continue
            if not pd.notna(score):
                horizon_skipped += 1
                continue

            record_prediction(
                symbol=str(symbol),
                horizon=horizon,
                timestamp=now,
                predicted_return=score,
                model_version=f"crypto_{horizon}@{meta.get('saved_at', 'unknown')}",
                feature_version="crypto-features-v1",
                price=float(row["close"]),
                source_mode=mode,
                data_quality="real_exchange_candle_fresh",
                data_cutoff_timestamp=cutoff.to_pydatetime(),
            )
            horizon_created += 1

        created += horizon_created
        report["by_horizon"][horizon] = {
            "recorded": horizon_created,
            "skipped_stale_or_invalid": horizon_skipped,
            "model_provenance": model_provenance,
        }

    report["models"] = provenance_report
    report["created_or_already_recorded"] = created
    report["status"] = "ok" if created > 0 else "degraded_no_fresh_real_forecasts"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_report(report)
    print(json.dumps(report, indent=2, default=str))
    return 0 if created > 0 else 4


def _write_report(report: dict) -> None:
    path = DATA / "reports" / "forecast_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str))
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
