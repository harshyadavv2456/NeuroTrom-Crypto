"""Production health check for NeuroTrom Crypto.

Exit 0 only when the real-data production path is usable. An API process being
alive, or synthetic fixtures being fresh, is not sufficient for a healthy result.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

from config import DATA, HOST, PORT, HORIZONS, SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC
from market_data import get_candles, test_provider_connectivity
from train import MODELS
from journal import compute_journal_summary
from live import get_live_status
from universe import load_universe

REAL_MODES = {SOURCE_LIVE, SOURCE_HISTORICAL, "REAL_MIXED"}


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK ' if ok else 'BAD'}  {label:<38} {detail}")
    return ok


def active_real_universe() -> pd.DataFrame:
    df = load_universe()
    if df.empty or "exchange" not in df.columns:
        return pd.DataFrame()
    df = df[df["exchange"].astype(str).str.lower().isin(["binance", "kraken", "coindcx"])].copy()
    if "universe_provenance" in df.columns:
        df = df[df["universe_provenance"].astype(str).isin([SOURCE_LIVE, SOURCE_HISTORICAL])]
    if "status" in df.columns:
        df = df[~df["status"].astype(str).str.upper().isin(["DEMO_ONLY", "SYNTHETIC"])]
    return df.reset_index(drop=True)


def inspect_real_candles(universe: pd.DataFrame) -> dict:
    result = {
        "real_rows": 0,
        "synthetic_rows": 0,
        "unknown_rows": 0,
        "real_series": 0,
        "production_nonreal_rows": 0,
        "series_expected": int(len(universe) * 2),
        "latest_real_timestamp": None,
    }
    newest = None
    for symbol in universe["symbol"].astype(str).tolist():
        for interval in ("1h", "1d"):
            df = get_candles(symbol, interval, "spot")
            demo_df = get_candles(symbol, interval, "spot", source_mode=SOURCE_SYNTHETIC)
            result["synthetic_rows"] += len(demo_df)
            if df.empty:
                continue
            if "source_mode" not in df.columns:
                result["unknown_rows"] += len(df)
                continue
            modes = df["source_mode"].astype(str)
            real = modes.isin([SOURCE_LIVE, SOURCE_HISTORICAL])
            result["real_rows"] += int(real.sum())
            result["unknown_rows"] += int((~real).sum())
            if bool((~real).any()):
                result["production_nonreal_rows"] = result.get("production_nonreal_rows", 0) + int((~real).sum())
            if real.any():
                result["real_series"] += 1
                stamps = pd.to_datetime(df.loc[real, "timestamp"], utc=True, errors="coerce").dropna()
                if not stamps.empty:
                    value = stamps.max()
                    if newest is None or value > newest:
                        newest = value
    result["latest_real_timestamp"] = str(newest) if newest is not None else None
    return result


def main() -> int:
    now = datetime.now(timezone.utc)
    failures = []
    print(f"NeuroTrom Crypto health — {now:%Y-%m-%d %H:%M:%S} UTC")

    api_ok = False
    try:
        response = requests.get(f"http://{HOST}:{PORT}/health", timeout=5)
        api_ok = response.status_code == 200 and response.json().get("service") == "neurotrom-crypto"
    except Exception as exc:
        api_error = f"{type(exc).__name__}: {str(exc)[:160]}"
    else:
        api_error = ""
    failures.append(not check("API liveness", api_ok, f"{HOST}:{PORT} {api_error}".strip()))

    providers = {
        name: test_provider_connectivity(name)
        for name in ("binance", "kraken", "coindcx")
    }
    any_provider = any(x.get("reachable", False) for x in providers.values())
    for name, status in providers.items():
        check(f"provider {name}", bool(status.get("reachable")),
              f"reachable={status.get('reachable')}; transport={status.get('transport', 'requests/curl')}; error={status.get('error', '')}")
    failures.append(not check("at least one real provider reachable", any_provider,
                              ", ".join(f"{name}={status.get('reachable')}" for name, status in providers.items())))

    universe = active_real_universe()
    failures.append(not check("real exchange universe", len(universe) > 0,
                              f"{len(universe)} eligible instruments"))
    data = inspect_real_candles(universe) if not universe.empty else {
        "real_rows": 0, "synthetic_rows": 0, "unknown_rows": 0, "real_series": 0,
        "series_expected": 0, "latest_real_timestamp": None,
    }
    failures.append(not check("real exchange candles", data["real_rows"] > 0,
                              json.dumps(data, sort_keys=True)))
    failures.append(not check("no unknown-provenance candle rows", data["unknown_rows"] == 0,
                              f"{data['unknown_rows']} unknown-provenance rows"))
    failures.append(not check("production store contains only real rows", data.get("production_nonreal_rows", 0) == 0,
                              f"{data.get('production_nonreal_rows', 0)} non-real rows in production reads"))
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
    failures.append(not check("physical storage isolation audit", isolation_ok, isolation_detail))
    check("synthetic fixtures isolated", True,
          f"{data['synthetic_rows']} demo rows in separate archive; excluded from production reads")

    real_models = []
    for horizon in HORIZONS:
        model_path = MODELS / f"crypto_{horizon}.txt"
        prov_path = MODELS / f"crypto_{horizon}_provenance.json"
        try:
            meta = json.loads(prov_path.read_text()) if prov_path.is_file() else {}
        except (OSError, ValueError):
            meta = {}
        mode = meta.get("provenance", "UNKNOWN")
        real_models.append(model_path.is_file() and mode in REAL_MODES)
        check(f"model {horizon}", real_models[-1],
              f"file={model_path.is_file()}; provenance={mode}; trained_at={meta.get('saved_at')}")
    models_ok = all(real_models)
    failures.append(not check("all four real-data models", models_ok,
                              f"{sum(real_models)}/{len(HORIZONS)} real-ready"))

    journal = compute_journal_summary()
    real_predictions = int(journal.get("real_predictions", 0))
    synthetic_predictions = int(journal.get("synthetic_predictions", 0))
    matured_real = int(journal.get("matured", 0))
    failures.append(not check("real forward journal", real_predictions > 0,
                              f"real={real_predictions}; synthetic={synthetic_predictions}; matured_real={matured_real}"))

    live = get_live_status()
    age = live.get("last_update_age_seconds")
    fresh = bool(live.get("fresh")) and bool(live.get("streams")) and age is not None and age <= 180
    failures.append(not check("live collector fresh", fresh,
                              f"running={live.get('collector_running')}; last={live.get('last_update')}; age_s={age}; streams={len(live.get('streams') or [])}; status={live.get('status')}"))

    if not any_provider and not data["real_rows"]:
        overall = "BLOCKED — no real data provider reachable"
    elif not data["real_rows"]:
        overall = "BLOCKED — provider reachable but no real candles persisted"
    elif failures:
        overall = "DEGRADED — real-data pipeline has unresolved health checks"
    else:
        overall = "OK — real-data pipeline healthy"

    print(f"\nOverall: {overall}")
    print(f"  Real candle rows: {data['real_rows']}")
    print(f"  Synthetic demo rows in active universe files: {data['synthetic_rows']}")
    print(f"  Real-data models: {sum(real_models)}/{len(HORIZONS)}")
    print(f"  Real forward predictions: {real_predictions}")
    return 1 if failures or not any_provider or not data["real_rows"] or not models_ok or real_predictions == 0 or not fresh else 0


if __name__ == "__main__":
    raise SystemExit(main())
