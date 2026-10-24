"""One-command, real-data-only bootstrap for NeuroTrom Crypto.

This command refreshes a real exchange universe, backfills all history the chosen
provider actually exposes, trains strictly from exchange data, and writes first
forward predictions only when all four models and fresh real candles are verified.
It deliberately never falls back to synthetic data.
"""
from __future__ import annotations

import fcntl
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import DATA, HORIZONS, SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC, UNIVERSE_SIZE
from market_data import test_provider_connectivity, get_candles
from universe import build_universe, load_universe
from backfill import backfill_universe
from features import build_feature_dataset
from train import run_training, predict, MODELS
from journal import record_prediction, compute_journal_summary

log = logging.getLogger(__name__)
REPORT = DATA / "reports" / "real_bootstrap.json"
REAL_MODES = {SOURCE_LIVE, SOURCE_HISTORICAL}


def _real_universe() -> pd.DataFrame:
    universe = load_universe()
    if universe.empty or "exchange" not in universe.columns:
        return pd.DataFrame()
    real = universe[universe["exchange"].astype(str).str.lower().isin(["binance", "kraken", "coindcx"])].copy()
    if "universe_provenance" in real.columns:
        real = real[real["universe_provenance"].astype(str).isin(REAL_MODES)]
    if "status" in real.columns:
        real = real[~real["status"].astype(str).str.upper().isin(["DEMO_ONLY", "SYNTHETIC"])]
    return real.reset_index(drop=True)


def _write_report(report: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str))
    tmp.replace(REPORT)


def _main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    result: dict = {
        "started_at": started,
        "real_data_only": True,
        "stages": {},
        "predictions_created": 0,
    }

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    providers = {
        name: test_provider_connectivity(name)
        for name in ("binance", "kraken", "coindcx")
    }
    result["stages"]["provider_connectivity"] = providers
    reachable = [name for name, info in providers.items() if info.get("reachable")]
    if not reachable:
        result["overall"] = "BLOCKED — no real data provider reachable"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(result)
        print(json.dumps(result, indent=2, default=str))
        return 2

    # Refresh the universe from official exchange metadata. If discovery fails,
    # retain an existing real universe but never promote the synthetic fixture list.
    try:
        preferred_exchange = next((name for name in ("binance", "kraken", "coindcx")
                                   if providers.get(name, {}).get("reachable")), "binance")
        build_universe(UNIVERSE_SIZE, preferred_exchange=preferred_exchange)
    except Exception as exc:
        log.warning("Universe refresh failed: %s", exc)
    universe = _real_universe()
    if universe.empty:
        result["overall"] = "BLOCKED — providers answer, but no real exchange universe could be resolved"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(result)
        print(json.dumps(result, indent=2, default=str))
        return 3
    result["stages"]["universe"] = {
        "count": int(len(universe)),
        "providers": sorted(set(universe["exchange"].astype(str).str.lower())),
        "symbols": universe["symbol"].astype(str).tolist(),
    }

    # Historical providers may limit what they expose. Store actual timestamps and
    # report coverage honestly rather than claiming the requested start was met.
    try:
        rows_written, errors = backfill_universe(
            intervals=["1h", "1d"],
            universe_limit=len(universe),
            days=365,
        )
    except Exception as exc:
        rows_written, errors = 0, [f"{type(exc).__name__}: {exc}"]
    result["stages"]["backfill"] = {
        "rows_written_or_upgraded": int(rows_written),
        "errors": errors[:100],
    }

    real_rows = 0
    series = {}
    for symbol in universe["symbol"].astype(str):
        for interval in ("1h", "1d"):
            df = get_candles(symbol, interval, "spot")
            if not df.empty and "source_mode" in df.columns:
                real = df[df["source_mode"].astype(str).isin(REAL_MODES)]
            else:
                real = pd.DataFrame()
            series[f"{symbol}_{interval}"] = {
                "real_rows": int(len(real)),
                "first": str(real["timestamp"].min()) if not real.empty else None,
                "last": str(real["timestamp"].max()) if not real.empty else None,
                "providers": sorted(set(real["provider"].astype(str))) if not real.empty and "provider" in real.columns else [],
            }
            real_rows += int(len(real))
    result["stages"]["stored_real_data"] = {
        "total_real_rows_across_series": real_rows,
        "series": series,
    }
    if real_rows == 0:
        result["overall"] = "BLOCKED — no real candles persisted"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(result)
        print(json.dumps(result, indent=2, default=str))
        return 4

    features = build_feature_dataset(interval="1h", require_real=True)
    feature_modes = set(features["source_mode"].astype(str).unique()) if not features.empty and "source_mode" in features.columns else set()
    if features.empty or not feature_modes or not feature_modes.issubset(REAL_MODES):
        result["stages"]["features"] = {
            "ok": False,
            "rows": int(len(features)),
            "provenance": sorted(feature_modes),
            "error": "No clean exchange-only feature dataset",
        }
        result["overall"] = "BLOCKED — insufficient verified real feature rows"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(result)
        print(json.dumps(result, indent=2, default=str))
        return 5

    result["stages"]["features"] = {
        "ok": True,
        "rows": int(len(features)),
        "symbols": int(features["symbol"].nunique()),
        "timestamps": int(features["timestamp"].nunique()),
        "provenance": sorted(feature_modes),
    }

    train_report = run_training(interval="1h", allow_synthetic=False)
    result["stages"]["training"] = train_report

    verified_models = {}
    for horizon in HORIZONS:
        model_file = MODELS / f"crypto_{horizon}.txt"
        provenance_file = MODELS / f"crypto_{horizon}_provenance.json"
        try:
            provenance = json.loads(provenance_file.read_text()) if provenance_file.exists() else {}
        except (OSError, ValueError):
            provenance = {}
        mode = provenance.get("provenance", "UNKNOWN")
        verified_models[horizon] = {
            "file_exists": model_file.is_file(),
            "provenance": mode,
            "trained_at": provenance.get("saved_at"),
            "real_ready": model_file.is_file() and mode in REAL_MODES | {"REAL_MIXED"},
        }
    result["stages"]["verified_models"] = verified_models
    all_models_real = all(v["real_ready"] for v in verified_models.values())
    if not all_models_real:
        result["overall"] = "DEGRADED — real ingestion works, but not all four real-data models passed"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(result)
        print(json.dumps(result, indent=2, default=str))
        return 6

    # Forward predictions are recorded at creation time with a separate input
    # candle cutoff. No historical backtest prediction is written as a live forecast.
    created = 0
    stale_cutoffs = []
    now = datetime.now(timezone.utc)
    for horizon in HORIZONS:
        prediction_scores = predict(features, horizon, allow_synthetic=False)
        if prediction_scores is None:
            continue
        meta = json.loads((MODELS / f"crypto_{horizon}_provenance.json").read_text())
        model_version = f"crypto_{horizon}@{meta.get('saved_at', 'unknown')}"
        for symbol in features["symbol"].astype(str).unique():
            symbol_rows = features[features["symbol"].astype(str) == symbol].sort_values("timestamp")
            if symbol_rows.empty:
                continue
            row = symbol_rows.iloc[-1]
            cutoff = pd.Timestamp(row["timestamp"])
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            age_hours = (pd.Timestamp.now(tz="UTC") - cutoff).total_seconds() / 3600
            source_mode = str(row.get("source_mode", "UNKNOWN"))
            if source_mode not in REAL_MODES or age_hours > 2.5:
                stale_cutoffs.append({"symbol": symbol, "horizon": horizon, "source_mode": source_mode, "age_hours": round(age_hours, 2)})
                continue
            score = float(prediction_scores.loc[row.name])
            record_prediction(
                symbol=symbol,
                horizon=horizon,
                timestamp=now,
                predicted_return=score,
                model_version=model_version,
                feature_version="crypto-features-v1",
                price=float(row["close"]),
                source_mode=source_mode,
                data_quality="exchange_data_fresh",
                data_cutoff_timestamp=cutoff.to_pydatetime(),
            )
            created += 1

    journal = compute_journal_summary()
    result["stages"]["forward_predictions"] = {
        "created_or_already_recorded": created,
        "stale_or_unverified_cutoffs": stale_cutoffs[:100],
        "journal": journal,
    }
    result["predictions_created"] = created
    result["overall"] = "REAL DATA READY — models trained and forward journal populated" if all_models_real and created > 0 else "DEGRADED — real data/models ready, but no fresh real forward predictions"
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_report(result)
    print(json.dumps(result, indent=2, default=str))
    return 0 if all_models_real and created > 0 else 7


def main() -> int:
    """Run at most one real-data bootstrap per machine at a time."""
    lock_path = DATA / "state" / "bootstrap_real.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("SKIP: another NeuroTrom Crypto real-data bootstrap is already running", flush=True)
            return 10
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
