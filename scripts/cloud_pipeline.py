#!/usr/bin/env python3
"""Cloud-only daily NeuroTrom Crypto pipeline.

Fetches public Binance Spot REST data, optionally extends the earliest history
from a license-compatible Kaggle dataset, trains non-production LightGBM research
models, and publishes compact JSON/CSV outputs for GitHub Pages. It does not use
CoinDCX data: CoinDCX's current API terms prohibit third-party display of its
market data and derived analytics.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.getenv("NEUROTROM_CLOUD_WORK", ROOT / ".cloud-work"))
SITE = ROOT / "site"
OUT = SITE / "data"
KAGGLE_DIR = WORK / "kaggle"
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT",
]
HORIZONS = {"1h": 1, "4h": 4, "24h": 24, "7d": 168}
BINANCE_HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api-gcp.binance.com",
]
INTERVAL_MS = 3_600_000
HISTORY_DAYS = int(os.getenv("NEUROTROM_HISTORY_DAYS", "365"))
KAGGLE_REF = os.getenv("KAGGLE_INPUT_DATASET", "").strip()
KAGGLE_STALE_DAYS = int(os.getenv("KAGGLE_STALE_AFTER_DAYS", "7"))
USER_AGENT = "NeuroTrom-Crypto-Research/1.0"
LOG = logging.getLogger("neurotrom.cloud")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        t = pd.Timestamp(value)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        return t.isoformat()
    except Exception:
        return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _request_json(url: str, params: dict | None = None, timeout: float = 8) -> Any:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    r = requests.get(url, params=params, timeout=(3, timeout), headers=headers)
    r.raise_for_status()
    return r.json()


def choose_binance_host() -> tuple[str | None, list[dict]]:
    checks = []
    for host in BINANCE_HOSTS:
        started = time.monotonic()
        try:
            _request_json(f"{host}/api/v3/ping", timeout=4)
            checks.append({"host": host, "reachable": True, "latency_ms": round((time.monotonic() - started) * 1000, 1)})
            return host, checks
        except Exception as exc:
            checks.append({"host": host, "reachable": False, "error": f"{type(exc).__name__}: {str(exc)[:180]}"})
    return None, checks


def get_with_host_failover(path: str, params: dict | None, base_host: str) -> Any:
    errors = []
    hosts = [base_host] + [h for h in BINANCE_HOSTS if h != base_host]
    for host in hosts:
        try:
            return _request_json(f"{host}{path}", params=params, timeout=12)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.SSLError) as exc:
            errors.append(f"{host}: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        except requests.HTTPError as exc:
            # A server response such as 400/403/429 is not a transport failover signal.
            raise RuntimeError(f"Binance returned HTTP error for {path}: {exc}") from exc
    raise RuntimeError("All supported Binance hosts failed: " + " | ".join(errors))


def fetch_ticker(symbol: str, host: str) -> dict:
    raw = get_with_host_failover("/api/v3/ticker/24hr", {"symbol": symbol}, host)
    return {
        "symbol": symbol,
        "price": float(raw["lastPrice"]),
        "change_24h_pct": float(raw["priceChangePercent"]),
        "high_24h": float(raw["highPrice"]),
        "low_24h": float(raw["lowPrice"]),
        "volume_24h": float(raw["volume"]),
        "quote_volume_24h": float(raw["quoteVolume"]),
        "source": "Binance Spot public REST",
        "timestamp": utc_now().isoformat(),
    }


def fetch_hourly_candles(symbol: str, host: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    end_ms = int(utc_now().timestamp() * 1000)
    start_ms = int((utc_now() - timedelta(days=days)).timestamp() * 1000)
    cursor = start_ms
    records: list[dict] = []
    seen = set()
    max_pages = int(days * 24 / 900) + 10
    for _ in range(max_pages):
        if cursor >= end_ms:
            break
        raw = get_with_host_failover(
            "/api/v3/klines",
            {"symbol": symbol, "interval": "1h", "startTime": cursor, "endTime": end_ms, "limit": 1000},
            host,
        )
        if not raw:
            break
        for row in raw:
            open_ms = int(row[0])
            if open_ms in seen:
                continue
            seen.add(open_ms)
            close_ms = int(row[6])
            # Exclude still-forming candles; a final partial bar is not a training label.
            if close_ms >= int(utc_now().timestamp() * 1000):
                continue
            records.append({
                "timestamp": pd.to_datetime(open_ms, unit="ms", utc=True),
                "open": float(row[1]), "high": float(row[2]), "low": float(row[3]),
                "close": float(row[4]), "volume": float(row[5]),
                "quote_volume": float(row[7]), "trades": int(row[8]),
                "symbol": symbol, "interval": "1h",
                "source": "BINANCE_PUBLIC_REST",
            })
        next_cursor = int(raw[-1][0]) + INTERVAL_MS
        if next_cursor <= cursor or len(raw) < 1:
            break
        cursor = next_cursor
        time.sleep(0.06)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).drop_duplicates("timestamp").sort_values("timestamp")
    return df.reset_index(drop=True)


def parse_kaggle_updated(meta: Any) -> datetime | None:
    for key in ("lastUpdated", "last_updated", "lastUpdatedAt", "last_updated_at"):
        value = getattr(meta, key, None)
        if value:
            try:
                return pd.Timestamp(value).tz_localize("UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC")
            except Exception:
                continue
    return None


def load_kaggle_source() -> tuple[dict, dict[str, pd.DataFrame]]:
    """Download only configured Kaggle data and accept permissively licensed,
    compatible hourly OHLCV rows for the same exact USDT symbol. Kaggle is optional.
    """
    status: dict = {
        "configured": bool(KAGGLE_REF),
        "dataset_ref": KAGGLE_REF or None,
        "status": "not_configured" if not KAGGLE_REF else "not_checked",
        "used_for_training": False,
        "rows_used": 0,
        "reason": None,
    }
    if not KAGGLE_REF:
        return status, {}
    has_token = bool(os.getenv("KAGGLE_API_TOKEN"))
    has_legacy_credentials = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    if not (has_token or has_legacy_credentials):
        status.update(status="not_configured", reason="Kaggle dataset ref is set but repository Kaggle credentials are missing.")
        return status, {}
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        meta = api.dataset_view(KAGGLE_REF)
        license_name = str(getattr(meta, "licenseName", "") or getattr(meta, "license_name", "") or "")
        updated_at = parse_kaggle_updated(meta)
        status["license"] = license_name or "unknown"
        status["last_updated"] = iso(updated_at)
        age_days = (utc_now() - updated_at.to_pydatetime()).total_seconds() / 86400 if updated_at else None
        status["age_days"] = round(max(0, age_days), 2) if age_days is not None else None
        status["freshness"] = "fresh" if age_days is not None and age_days <= KAGGLE_STALE_DAYS else ("stale" if age_days is not None else "unknown")
        license_upper = license_name.upper()
        license_ok = (
            "CC0" in license_upper
            or "PUBLIC DOMAIN" in license_upper
            or "CC BY 4.0" in license_upper
            or "CC-BY-4.0" in license_upper
        )
        status["license_accepted"] = license_ok
        if not license_ok:
            status.update(status="skipped", reason="Dataset license is not in the pipeline's explicit permissive-license allowlist; no files were downloaded or used.")
            return status, {}
        KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
        # Clear old downloaded files first; otherwise removed/renamed files from an
        # earlier dataset version can silently contaminate today's parsed input.
        for stale_path in sorted(KAGGLE_DIR.rglob("*"), reverse=True):
            if stale_path.is_file():
                stale_path.unlink()
            elif stale_path.is_dir():
                try:
                    stale_path.rmdir()
                except OSError:
                    pass
        api.dataset_download_files(KAGGLE_REF, path=str(KAGGLE_DIR), unzip=True, quiet=True)
        frames = parse_compatible_kaggle_csvs(KAGGLE_DIR)
        rows = sum(len(frame) for frame in frames.values())
        latest_candle = max((frame["timestamp"].max() for frame in frames.values() if not frame.empty), default=None)
        candle_age_days = ((utc_now() - pd.Timestamp(latest_candle).to_pydatetime()).total_seconds() / 86400) if latest_candle is not None else None
        data_freshness = "fresh" if candle_age_days is not None and 0 <= candle_age_days <= KAGGLE_STALE_DAYS else ("stale" if candle_age_days is not None else "unknown")
        status["dataset_metadata_freshness"] = status.get("freshness", "unknown")
        status["latest_candle_timestamp"] = iso(latest_candle)
        status["latest_candle_age_days"] = round(max(0, candle_age_days), 2) if candle_age_days is not None else None
        status["data_freshness"] = data_freshness
        # The UI's main Kaggle freshness label is based on actual rows, not just
        # metadata being edited recently. Old rows may still extend old history,
        # but are never treated as current prices.
        status["freshness"] = data_freshness
        status["rows_compatible"] = rows
        status["rows_used"] = 0
        status["used_for_training"] = False
        status["status"] = "loaded" if rows else "skipped"
        if not rows:
            status["reason"] = "No exact-symbol hourly USDT OHLCV files matched the supported schema; not mixing daily, USD, or ambiguous candles."
        elif data_freshness == "stale":
            status["reason"] = f"Dataset candle rows end {status['latest_candle_timestamp']} ({status['latest_candle_age_days']} days old). Compatible rows may extend historical coverage only; current prices come from the live public API."
        return status, frames
    except Exception as exc:
        status.update(status="error", reason=f"{type(exc).__name__}: {str(exc)[:260]}")
        LOG.warning("Kaggle source unavailable; continuing with live exchange data: %s", exc)
        return status, {}


def _normalise_columns(df: pd.DataFrame) -> dict[str, str]:
    names = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    candidates = {
        "timestamp": ("timestamp", "opentime", "datetime", "date", "time", "unix", "dateutc"),
        "open": ("open", "openprice", "o"),
        "high": ("high", "highprice", "h"),
        "low": ("low", "lowprice", "l"),
        "close": ("close", "closeprice", "price", "last", "c"),
        "volume": ("volume", "basevolume", "vol", "v"),
        "symbol": ("symbol", "ticker", "pair", "market", "coin"),
    }
    found = {}
    for target, options in candidates.items():
        for option in options:
            if option in names:
                found[target] = names[option]
                break
    return found


def _infer_symbol(value: str) -> str | None:
    text = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    for symbol in SYMBOLS:
        if symbol in text:
            return symbol
    return None


def parse_compatible_kaggle_csvs(folder: Path) -> dict[str, pd.DataFrame]:
    accepted: dict[str, list[pd.DataFrame]] = {}
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".csv" or path.name.lower() in {"daily_update.csv"}:
            continue
        try:
            raw = pd.read_csv(path, low_memory=False)
            cols = _normalise_columns(raw)
            required = {"timestamp", "open", "high", "low", "close", "volume"}
            if not required.issubset(cols):
                continue
            if "symbol" in cols:
                symbols = raw[cols["symbol"]].astype(str).map(_infer_symbol)
            else:
                inferred = _infer_symbol(path.stem)
                if not inferred:
                    continue
                symbols = pd.Series([inferred] * len(raw), index=raw.index)
            ts = raw[cols["timestamp"]]
            if pd.api.types.is_numeric_dtype(ts):
                numeric = pd.to_numeric(ts, errors="coerce")
                median = numeric.dropna().abs().median()
                unit = "ms" if median > 1e11 else "s" if median > 1e8 else None
                if not unit:
                    continue
                timestamps = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
            else:
                timestamps = pd.to_datetime(ts, utc=True, errors="coerce")
            frame = pd.DataFrame({
                "timestamp": timestamps,
                "open": pd.to_numeric(raw[cols["open"]], errors="coerce"),
                "high": pd.to_numeric(raw[cols["high"]], errors="coerce"),
                "low": pd.to_numeric(raw[cols["low"]], errors="coerce"),
                "close": pd.to_numeric(raw[cols["close"]], errors="coerce"),
                "volume": pd.to_numeric(raw[cols["volume"]], errors="coerce"),
                "symbol": symbols,
            }).dropna(subset=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
            if frame.empty:
                continue
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            diffs = frame.sort_values("timestamp").groupby("symbol")["timestamp"].diff().dropna().dt.total_seconds()
            median_hours = diffs.median() / 3600 if not diffs.empty else None
            # Strictly accept hourly data only; daily CSVs are not upsampled into fake candles.
            if median_hours is None or not 0.75 <= median_hours <= 1.5:
                continue
            frame["interval"] = "1h"
            frame["source"] = f"KAGGLE:{path.name}"
            for symbol, group in frame.groupby("symbol"):
                accepted.setdefault(str(symbol), []).append(group)
        except Exception as exc:
            LOG.warning("Skipping unparseable Kaggle file %s: %s", path.name, exc)
    result = {}
    for symbol, parts in accepted.items():
        result[symbol] = (
            pd.concat(parts, ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    return result


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.sort_values("timestamp").copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").clip(lower=0)
    log_close = np.log(close.where(close > 0))
    features = pd.DataFrame(index=df.index)
    for lag in (1, 2, 4, 8, 24, 48, 168):
        features[f"ret_{lag}"] = close.pct_change(lag)
        features[f"logret_{lag}"] = log_close.diff(lag)
    for window in (12, 24, 72, 168):
        features[f"volatility_{window}"] = log_close.diff().rolling(window).std()
        features[f"ma_gap_{window}"] = close / close.rolling(window).mean() - 1
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    features["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    features["range_pct"] = (df["high"] - df["low"]) / close.replace(0, np.nan)
    features["body_pct"] = (df["close"] - df["open"]) / close.replace(0, np.nan)
    features["volume_log_change"] = np.log1p(volume).diff()
    features["hour_utc"] = pd.to_datetime(df["timestamp"], utc=True).dt.hour
    features["day_utc"] = pd.to_datetime(df["timestamp"], utc=True).dt.dayofweek
    features["symbol"] = df["symbol"].astype(str)
    return features.replace([np.inf, -np.inf], np.nan)


def train_models(data: dict[str, pd.DataFrame], tickers: dict[str, dict]) -> tuple[dict, list[dict]]:
    training_rows = []
    latest_rows = []
    for symbol, frame in data.items():
        if frame.empty:
            continue
        frame = frame.drop_duplicates("timestamp").sort_values("timestamp").copy()
        features = add_features(frame)
        for horizon, bars in HORIZONS.items():
            features[f"label_{horizon}"] = frame["close"].shift(-bars) / frame["close"] - 1
        combined = frame[["timestamp", "symbol", "close", "source"]].join(features.drop(columns=["symbol"]))
        training_rows.append(combined)
        valid = features.drop(columns=[f"label_{h}" for h in HORIZONS]).dropna()
        if not valid.empty:
            latest_rows.append((symbol, frame, valid.iloc[[-1]]))
    if not training_rows:
        return {"status": "blocked", "models": {}, "reason": "No historical hourly market data was fetched."}, []
    all_rows = pd.concat(training_rows, ignore_index=True)
    numeric_features = [c for c in all_rows.columns if c not in {"timestamp", "symbol", "close", "source"} and not c.startswith("label_")]
    model_report = {"status": "ok", "models": {}, "training_rows": int(len(all_rows)), "features": numeric_features}
    predictions = []
    models_ready = 0
    for horizon, bars in HORIZONS.items():
        label = f"label_{horizon}"
        eligible = all_rows.dropna(subset=[label] + numeric_features).copy()
        eligible = eligible.sort_values(["timestamp", "symbol"])
        times = pd.Index(eligible["timestamp"].drop_duplicates().sort_values())
        if len(eligible) < 500 or len(times) < max(400, bars * 3):
            model_report["models"][horizon] = {
                "status": "insufficient_data", "rows": int(len(eligible)),
                "note": "Insufficient clean hourly history for a reliable holdout.",
            }
            continue
        cutoff_index = int(len(times) * 0.8)
        if cutoff_index <= bars + 1 or cutoff_index >= len(times):
            model_report["models"][horizon] = {"status": "insufficient_data", "rows": int(len(eligible))}
            continue
        cutoff = times[cutoff_index]
        # Embargo the entire forecast horizon from training so the final train
        # labels cannot observe any future close inside the holdout window.
        train_end = times[cutoff_index - bars]
        train = eligible[eligible["timestamp"] < train_end]
        test = eligible[eligible["timestamp"] >= cutoff]
        if len(train) < 300 or len(test) < 100:
            model_report["models"][horizon] = {"status": "insufficient_data", "rows": int(len(eligible))}
            continue
        params = {
            "objective": "regression", "n_estimators": 220, "learning_rate": 0.03,
            "num_leaves": 15, "max_depth": -1, "min_child_samples": 80,
            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.75, "reg_lambda": 3.0,
            "random_state": 42, "verbosity": -1, "n_jobs": 2,
        }
        try:
            validation_model = lgb.LGBMRegressor(**params)
            validation_model.fit(train[numeric_features], train[label])
            pred = validation_model.predict(test[numeric_features])
            ic = spearmanr(pred, test[label], nan_policy="omit").statistic
            mae = float(np.mean(np.abs(pred - test[label].to_numpy())))
            directional = float(np.mean(np.sign(pred) == np.sign(test[label].to_numpy())))
            model = lgb.LGBMRegressor(**params)
            model.fit(eligible[numeric_features], eligible[label])
            models_ready += 1
            model_report["models"][horizon] = {
                "status": "trained", "provenance": "BINANCE_PUBLIC_REST",
                "training_rows": int(len(eligible)), "validation_rows": int(len(test)),
                "validation_cutoff": iso(cutoff), "training_end_exclusive": iso(train_end),
                "embargo_hours": bars,
                "holdout_spearman_ic": round(float(ic), 5) if np.isfinite(ic) else None,
                "holdout_mae_return": round(mae, 6), "directional_accuracy": round(directional, 4),
                "n_estimators": params["n_estimators"],
                "metric_note": "Time-ordered holdout metrics; not proof of profitability or future performance.",
            }
            for symbol, frame, feature_row in latest_rows:
                current_ticker = tickers.get(symbol)
                if current_ticker is None:
                    continue
                try:
                    predicted_return = float(model.predict(feature_row[numeric_features])[0])
                except Exception:
                    continue
                if not np.isfinite(predicted_return):
                    continue
                # The model return is defined relative to the completed candle
                # close used to construct its features, not the later live ticker.
                reference_close = float(frame["close"].iloc[-1])
                predicted_at = utc_now().isoformat()
                predictions.append({
                    "journal_id": f"{predicted_at}:{symbol}:{horizon}:{os.getenv('GITHUB_RUN_ID', 'local')}",
                    "run_id": os.getenv("GITHUB_RUN_ID", "local"),
                    "symbol": symbol, "horizon": horizon,
                    "last_price": reference_close,
                    "current_ticker_price": float(current_ticker["price"]),
                    "entry_price": reference_close,
                    "predicted_return_pct": round(predicted_return * 100, 3),
                    "implied_target_price": round(max(0.0, reference_close * (1 + predicted_return)), 8),
                    "input_candle_time": iso(frame["timestamp"].iloc[-1]),
                    "forecasted_at": predicted_at,
                    "input_age_hours": round(max(0, (utc_now() - pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime()).total_seconds() / 3600), 2),
                    "data_source": "Binance Spot public REST",
                    "model_provenance": "BINANCE_PUBLIC_REST",
                    "matured": False,
                })
        except Exception as exc:
            model_report["models"][horizon] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:220]}"}
    model_report["ready_count"] = models_ready
    model_report["status"] = "ok" if models_ready == len(HORIZONS) else "degraded"
    return model_report, predictions


def load_public_journal() -> list[dict]:
    """Recover prior daily forecast records from the currently deployed Pages site."""
    repository = os.getenv("GITHUB_REPOSITORY", "harshyadavv2456/NeuroTrom-Crypto")
    try:
        owner, name = repository.split("/", 1)
        url = f"https://{owner}.github.io/{name}/data/journal.json"
        response = requests.get(url, params={"cache": str(int(time.time()))}, timeout=(2, 5),
                                headers={"User-Agent": USER_AGENT, "Cache-Control": "no-cache"})
        if response.status_code != 200:
            return []
        payload = response.json()
        records = payload.get("predictions", []) if isinstance(payload, dict) else []
        return [x for x in records if isinstance(x, dict) and x.get("journal_id")]
    except Exception as exc:
        LOG.info("No previous public journal loaded: %s", str(exc)[:180])
        return []


def update_journal(previous: list[dict], new_predictions: list[dict],
                   data: dict[str, pd.DataFrame]) -> tuple[list[dict], dict]:
    """Mature old forecasts against later completed candles, then append this run."""
    records = [dict(x) for x in previous]
    by_id = {str(x.get("journal_id")): i for i, x in enumerate(records)}
    for item in records:
        if item.get("matured"):
            continue
        symbol = str(item.get("symbol", ""))
        horizon = str(item.get("horizon", ""))
        frame = data.get(symbol)
        if frame is None or frame.empty or horizon not in HORIZONS:
            continue
        try:
            input_time = pd.Timestamp(item["input_candle_time"])
            if input_time.tzinfo is None:
                input_time = input_time.tz_localize("UTC")
            target_time = input_time + pd.Timedelta(hours=HORIZONS[horizon])
            eligible = frame[frame["timestamp"] >= target_time].sort_values("timestamp")
            if eligible.empty:
                continue
            actual = eligible.iloc[0]
            entry = float(item.get("entry_price", item["last_price"]))
            actual_close = float(actual["close"])
            if entry <= 0 or not np.isfinite(actual_close):
                continue
            realized = actual_close / entry - 1
            predicted = float(item["predicted_return_pct"]) / 100
            item.update({
                "matured": True,
                "matured_at": utc_now().isoformat(),
                "actual_candle_time": iso(actual["timestamp"]),
                "actual_close": actual_close,
                "actual_return_pct": round(realized * 100, 4),
                "directional_correct": bool(np.sign(predicted) == np.sign(realized)),
            })
        except Exception as exc:
            LOG.debug("Could not mature journal row: %s", str(exc)[:120])
    for item in new_predictions:
        key = str(item.get("journal_id", ""))
        if key and key not in by_id:
            by_id[key] = len(records)
            records.append(dict(item))
    records = records[-10000:]
    matured = [x for x in records if x.get("matured") and num_finite(x.get("actual_return_pct"))]
    by_horizon = {}
    for horizon in HORIZONS:
        subset = [x for x in matured if x.get("horizon") == horizon]
        by_horizon[horizon] = {
            "matured": len(subset),
            "directional_accuracy": round(sum(bool(x.get("directional_correct")) for x in subset) / len(subset), 4) if subset else None,
            "mean_actual_return_pct": round(float(np.mean([float(x["actual_return_pct"]) for x in subset])), 4) if subset else None,
        }
    summary = {
        "total_predictions": len(records),
        "matured_predictions": len(matured),
        "pending_predictions": len(records) - len(matured),
        "directional_accuracy": round(sum(bool(x.get("directional_correct")) for x in matured) / len(matured), 4) if matured else None,
        "mean_actual_return_pct": round(float(np.mean([float(x["actual_return_pct"]) for x in matured])), 4) if matured else None,
        "by_horizon": by_horizon,
    }
    return records, summary


def num_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def publish_outputs(tickers: dict, data: dict[str, pd.DataFrame], models: dict,
                    predictions: list[dict], kaggle_status: dict, host_checks: list[dict],
                    errors: list[str], journal: list[dict], journal_summary: dict) -> None:
    now = utc_now()
    latest_by_symbol = {}
    history_series = {}
    for symbol in SYMBOLS:
        frame = data.get(symbol, pd.DataFrame())
        ticker = tickers.get(symbol)
        if ticker:
            latest_by_symbol[symbol] = ticker
        if not frame.empty:
            recent = frame.sort_values("timestamp").tail(168)
            history_series[symbol] = [
                {"time": iso(row.timestamp), "close": round(float(row.close), 8)}
                for row in recent.itertuples()
            ]
    fresh_symbols = []
    for symbol, frame in data.items():
        if not frame.empty:
            age = (now - pd.Timestamp(frame["timestamp"].max()).to_pydatetime()).total_seconds() / 3600
            if age <= 2.5:
                fresh_symbols.append(symbol)
    ticker_symbols = sorted(latest_by_symbol)
    overall = "OK" if len(fresh_symbols) >= 3 and len(ticker_symbols) >= 3 and models.get("ready_count") == len(HORIZONS) else "DEGRADED"
    payload = {
        "project": "NeuroTrom Crypto Research",
        "generated_at": now.isoformat(),
        "run": {"commit": os.getenv("GITHUB_SHA", "local"), "run_id": os.getenv("GITHUB_RUN_ID"), "workflow": os.getenv("GITHUB_WORKFLOW")},
        "status": overall,
        "status_note": "Research-only. Historical model validation is not proof of profitability. No order execution.",
        "provider": {
            "name": "Binance Spot public REST",
            "host_checks": host_checks,
            "fresh_symbols": fresh_symbols,
            "ticker_symbols": ticker_symbols,
            "requested_symbols": SYMBOLS,
        },
        "kaggle": kaggle_status,
        "market": list(latest_by_symbol.values()),
        "models": models,
        "predictions": predictions,
        "journal": journal_summary,
        "errors": errors[-30:],
        "limits": {"historical_days_requested": HISTORY_DAYS, "symbols_requested": SYMBOLS, "history_series_points_per_symbol": 168},
    }
    write_json(OUT / "latest.json", payload)
    write_json(OUT / "history.json", {"generated_at": now.isoformat(), "interval": "1h", "series": history_series})
    write_json(OUT / "journal.json", {"updated_at": now.isoformat(), "predictions": journal})
    rows = []
    by_key = {(p["symbol"], p["horizon"]): p for p in predictions}
    for symbol in SYMBOLS:
        tick = tickers.get(symbol, {})
        for horizon in HORIZONS:
            prediction = by_key.get((symbol, horizon), {})
            rows.append({
                "generated_at": now.isoformat(),
                "symbol": symbol,
                "price": tick.get("price"),
                "change_24h_pct": tick.get("change_24h_pct"),
                "horizon": horizon,
                "predicted_return_pct": prediction.get("predicted_return_pct"),
                "implied_target_price": prediction.get("implied_target_price"),
                "input_candle_time": prediction.get("input_candle_time"),
                "model_status": models.get("models", {}).get(horizon, {}).get("status", "unknown"),
                "source": "Binance Spot public REST",
            })
    pd.DataFrame(rows).to_csv(OUT / "daily_update.csv", index=False)
    write_json(OUT / "run_history.json", {
        "latest_run": now.isoformat(), "status": overall,
        "commit": os.getenv("GITHUB_SHA", "local"),
        "models_ready": models.get("ready_count", 0),
        "kaggle_freshness": kaggle_status.get("freshness", kaggle_status.get("status")),
        "errors": errors[-30:],
    })


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = utc_now()
    errors: list[str] = []
    LOG.info("Cloud refresh started; history=%sd; configured Kaggle=%s", HISTORY_DAYS, bool(KAGGLE_REF))
    host, host_checks = choose_binance_host()
    data: dict[str, pd.DataFrame] = {}
    tickers: dict[str, dict] = {}
    kaggle_status, kaggle_frames = load_kaggle_source()

    previous_journal = load_public_journal()
    if host is None:
        errors.append("No Binance public REST host was reachable from this runner.")
        kaggle_status.setdefault("reason", "Live API host unreachable; dashboard will report degraded status.")
        models = {"status": "blocked", "models": {}, "ready_count": 0, "reason": errors[0]}
        journal, journal_summary = update_journal(previous_journal, [], data)
        publish_outputs(tickers, data, models, [], kaggle_status, host_checks, errors, journal, journal_summary)
        return 2

    for symbol in SYMBOLS:
        try:
            tickers[symbol] = fetch_ticker(symbol, host)
        except Exception as exc:
            errors.append(f"{symbol} ticker: {type(exc).__name__}: {str(exc)[:200]}")
        try:
            api_frame = fetch_hourly_candles(symbol, host)
            if api_frame.empty:
                raise RuntimeError("API returned no completed hourly candles")
            if symbol in kaggle_frames:
                old = kaggle_frames[symbol]
                cutoff = api_frame["timestamp"].min()
                old = old[old["timestamp"] < cutoff].copy()
                if not old.empty:
                    old["source"] = old["source"].astype(str)
                    api_frame = pd.concat([old, api_frame], ignore_index=True)
                    api_frame = api_frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
                    kaggle_status["rows_used"] = int(kaggle_status.get("rows_used", 0)) + int(len(old))
                    kaggle_status["used_for_training"] = True
            data[symbol] = api_frame
            LOG.info("%s: %d hourly bars, last=%s", symbol, len(api_frame), api_frame["timestamp"].max())
        except Exception as exc:
            errors.append(f"{symbol} candles: {type(exc).__name__}: {str(exc)[:220]}")
            if symbol in kaggle_frames and not kaggle_frames[symbol].empty:
                data[symbol] = kaggle_frames[symbol].copy()
                errors.append(f"{symbol}: using compatible historical Kaggle candles only; they are not current API data.")
    if kaggle_frames and not kaggle_status.get("used_for_training"):
        prior_reason = kaggle_status.get("reason")
        kaggle_status["reason"] = (
            "Compatible Kaggle rows were found, but none extend the history before the live API window; no Kaggle rows were used."
            + (f" Previous note: {prior_reason}" if prior_reason else "")
        )
    models, predictions = train_models(data, tickers)
    journal, journal_summary = update_journal(previous_journal, predictions, data)
    publish_outputs(tickers, data, models, predictions, kaggle_status, host_checks, errors, journal, journal_summary)
    runtime_s = round((utc_now() - started).total_seconds(), 1)
    LOG.info("Cloud refresh finished in %ss; live symbols=%s; models=%s; errors=%s",
             runtime_s, len(tickers), models.get("ready_count", 0), len(errors))
    return 0 if tickers and models.get("ready_count", 0) == len(HORIZONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
