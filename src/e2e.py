"""Non-destructive end-to-end verification of the real-data crypto pipeline.

This test never launches synthetic training, overwrites production models, performs
backfills, or fabricates predictions. It reports the real source/runtime state and
exits nonzero unless the end-to-end production path is actually ready.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from config import DATA, HOST, PORT, HORIZONS, SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC
from market_data import get_candles, test_provider_connectivity
from features import build_feature_dataset
from train import MODELS, _check_provenance, train_model
from journal import compute_journal_summary
from live import get_live_status
from universe import load_universe

log = logging.getLogger(__name__)
RESULTS = []
REPORTS_DIR = DATA / "reports"
REAL_MODES = {SOURCE_LIVE, SOURCE_HISTORICAL, "REAL_MIXED"}


def check(layer: str, name: str, fn):
    started = time.monotonic()
    try:
        result = fn() if callable(fn) else fn
        ok, detail = result if isinstance(result, tuple) else (True, str(result))
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {str(exc)[:250]}"
    RESULTS.append({
        "layer": layer, "name": name, "status": "PASS" if ok else "FAIL",
        "detail": str(detail)[:500], "seconds": round(time.monotonic() - started, 2),
    })
    print(f"{'PASS' if ok else 'FAIL'} [{layer}] {name} — {str(detail)[:180]}", flush=True)
    return bool(ok)


def _real_universe() -> pd.DataFrame:
    df = load_universe()
    if df.empty or "exchange" not in df.columns:
        return pd.DataFrame()
    real = df[df["exchange"].astype(str).str.lower().isin(["binance", "kraken", "coindcx"])].copy()
    if "universe_provenance" in real.columns:
        real = real[real["universe_provenance"].astype(str).isin([SOURCE_LIVE, SOURCE_HISTORICAL])]
    if "status" in real.columns:
        real = real[~real["status"].astype(str).str.upper().isin(["DEMO_ONLY", "SYNTHETIC"])]
    return real.reset_index(drop=True)


def _real_candle_counts(universe: pd.DataFrame) -> dict:
    real_rows = 0
    synthetic_rows = 0
    unknown_rows = 0
    real_series = 0
    latest = None
    for symbol in universe["symbol"].astype(str).tolist():
        for interval in ("1h", "1d"):
            df = get_candles(symbol, interval, "spot")
            demo_df = get_candles(symbol, interval, "spot", source_mode=SOURCE_SYNTHETIC)
            synthetic_rows += len(demo_df)
            if df.empty:
                continue
            if "source_mode" not in df.columns:
                unknown_rows += len(df)
                continue
            modes = df["source_mode"].astype(str)
            real_mask = modes.isin([SOURCE_LIVE, SOURCE_HISTORICAL])
            unknown_rows += int((~real_mask).sum())
            real_rows += int(real_mask.sum())
            if real_mask.any():
                real_series += 1
                stamp = pd.to_datetime(df.loc[real_mask, "timestamp"], utc=True).max()
                if pd.notna(stamp) and (latest is None or stamp > latest):
                    latest = stamp
    return {
        "real_rows": real_rows,
        "synthetic_rows": synthetic_rows,
        "unknown_rows": unknown_rows,
        "real_series": real_series,
        "series_expected": int(len(universe) * 2),
        "latest_real_candle": str(latest) if latest is not None else None,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = datetime.now(timezone.utc).isoformat()

    def provider_check(provider: str):
        result = test_provider_connectivity(provider)
        return ("provider" in result and isinstance(result.get("reachable"), bool),
                f"reachable={result.get('reachable')}; transport={result.get('transport', 'requests/curl')}; error={result.get('error', '')}")
    check("sources", "binance_connectivity_reported", lambda: provider_check("binance"))
    check("sources", "kraken_connectivity_reported", lambda: provider_check("kraken"))
    check("sources", "coindcx_connectivity_reported", lambda: provider_check("coindcx"))

    universe = _real_universe()
    check("universe", "real_exchange_universe_loaded", (len(universe) > 0, f"{len(universe)} real instruments"))
    data = _real_candle_counts(universe) if not universe.empty else {
        "real_rows": 0, "synthetic_rows": 0, "unknown_rows": 0, "real_series": 0,
        "series_expected": 0, "latest_real_candle": None,
    }
    check("data", "real_exchange_candles_persisted",
          (data["real_rows"] > 0, json.dumps(data, sort_keys=True)))
    check("data", "synthetic_archive_not_served_as_production",
          (data["unknown_rows"] == 0, f"archive rows={data['synthetic_rows']}; unknown production rows={data['unknown_rows']}"))
    migration_path = DATA / "reports" / "synthetic_migration.json"
    try:
        migration = json.loads(migration_path.read_text())
        audit = migration.get("production_store_audit", {})
        isolation_ok = (migration.get("status") == "ok"
                        and audit.get("active_synthetic_rows") == 0
                        and audit.get("active_unknown_rows") == 0)
        isolation_detail = (f"status={migration.get('status')}; "
                            f"active_synthetic={audit.get('active_synthetic_rows')}; "
                            f"active_unknown={audit.get('active_unknown_rows')}")
    except (OSError, ValueError, TypeError):
        isolation_ok, isolation_detail = False, "migration report missing or unreadable"
    check("safety", "physical_storage_isolation_verified", (isolation_ok, isolation_detail))

    features = build_feature_dataset(interval="1h", require_real=True)
    feature_modes = set(features["source_mode"].astype(str).unique()) if not features.empty and "source_mode" in features.columns else set()
    features_real = bool(len(features)) and bool(feature_modes) and feature_modes.issubset({SOURCE_LIVE, SOURCE_HISTORICAL})
    check("features", "production_features_are_exchange_only",
          (features_real, f"rows={len(features)} symbols={features['symbol'].nunique() if not features.empty else 0} provenance={sorted(feature_modes)}"))

    # Exercise the synthetic rejection before any fit begins. The test data is
    # in-memory only and must not be persisted to the production feature store.
    def synthetic_training_guard():
        fake = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"),
            "symbol": ["TESTCOIN"] * 200,
            "source_mode": [SOURCE_SYNTHETIC] * 200,
            "provider": ["synthetic"] * 200,
            "feature_a": range(200),
            "label_24h": [0.01] * 200,
        })
        try:
            train_model(fake, "24h", allow_synthetic=False)
        except ValueError as exc:
            return "synthetic" in str(exc).lower() or "provenance" in str(exc).lower(), str(exc)
        return False, "training accepted synthetic-only fixture without explicit opt-in"
    check("safety", "production_training_rejects_synthetic", synthetic_training_guard)

    model_rows = []
    for horizon in HORIZONS:
        model_file = MODELS / f"crypto_{horizon}.txt"
        provenance_file = MODELS / f"crypto_{horizon}_provenance.json"
        try:
            meta = json.loads(provenance_file.read_text()) if provenance_file.is_file() else {}
        except (OSError, ValueError):
            meta = {}
        provenance = meta.get("provenance", "UNKNOWN")
        model_rows.append({
            "horizon": horizon,
            "file_exists": model_file.is_file(),
            "provenance": provenance,
            "real_ready": model_file.is_file() and provenance in REAL_MODES,
        })
    models_real = all(row["real_ready"] for row in model_rows)
    check("models", "all_horizons_have_real_models", (models_real, json.dumps(model_rows, sort_keys=True)))

    journal = compute_journal_summary()
    real_predictions = int(journal.get("real_predictions", 0))
    check("journal", "forward_real_predictions_recorded",
          (real_predictions > 0, f"real={real_predictions}; synthetic={journal.get('synthetic_predictions', 0)}; matured_real={journal.get('matured', 0)}"))

    live = get_live_status()
    live_fresh = bool(live.get("fresh")) and bool(live.get("streams"))
    check("collector", "live_market_observations_fresh",
          (live_fresh, f"running={live.get('collector_running')}; fresh={live.get('fresh')}; last={live.get('last_update')}; streams={len(live.get('streams') or [])}; errors={len(live.get('errors') or [])}"))

    api_payload = {}
    api_ok = False
    try:
        response = requests.get(f"http://{HOST}:{PORT}/v1/crypto/status", timeout=15)
        response.raise_for_status()
        api_payload = response.json()
        actual_has_real = data["real_rows"] > 0
        api_ok = bool(api_payload.get("api_live")) and bool(api_payload.get("real_data_available")) == actual_has_real
        api_detail = f"overall={api_payload.get('overall')}; reported_real={api_payload.get('real_data_available')}; persisted_real={actual_has_real}"
    except Exception as exc:
        api_detail = f"{type(exc).__name__}: {exc}"
    check("api", "status_matches_persisted_data", (api_ok, api_detail))

    overall_ready = api_payload.get("overall") == "OK" and data["real_rows"] > 0 and models_real and real_predictions > 0 and live_fresh
    overall = "OK" if overall_ready and all(x["status"] == "PASS" for x in RESULTS) else str(api_payload.get("overall") or "BLOCKED — API status unavailable")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = [
        "# NeuroTrom Crypto End-to-End Report",
        "",
        f"- Started: {started}",
        f"- Finished: {datetime.now(timezone.utc).isoformat()}",
        f"- Overall: **{overall}**",
        f"- Real exchange universe: {len(universe)} instruments",
        f"- Real candle rows: {data['real_rows']}",
        f"- Synthetic rows in isolated demo archive: {data['synthetic_rows']}",
        f"- Unknown-provenance rows: {data['unknown_rows']}",
        f"- Real forward predictions: {real_predictions}",
        "",
        "| Layer | Check | Status | Detail |",
        "|---|---|---|---|",
    ]
    for item in RESULTS:
        detail = item["detail"].replace("|", "\\|").replace("\n", " ")
        report.append(f"| {item['layer']} | {item['name']} | {item['status']} | {detail} |")
    report.extend(["", "This report does not train models or write forecasts. It only inspects runtime state and uses an in-memory synthetic fixture to test the training guard."])
    (REPORTS_DIR / "e2e_report.md").write_text("\n".join(report) + "\n")

    passed = sum(item["status"] == "PASS" for item in RESULTS)
    failed = len(RESULTS) - passed
    print(f"\nE2E result: {passed}/{len(RESULTS)} checks passed; overall={overall}; report={REPORTS_DIR / 'e2e_report.md'}")
    return 0 if failed == 0 and overall_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
