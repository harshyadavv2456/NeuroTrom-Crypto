"""LightGBM training with walk-forward validation and provenance checks."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import DATA, HORIZONS, HORIZON_BARS, BASE_PARAMS, ROUNDS, N_FOLDS, GAP, COST_BPS, SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC
from features import build_feature_dataset, load_features, save_features

log = logging.getLogger(__name__)
MODELS = DATA / "models"
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS = DATA / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def _labels(df: pd.DataFrame, horizon: str) -> pd.Series:
    col = f"label_{horizon}"
    if col not in df.columns:
        raise ValueError(f"Missing label column: {col}")
    return df[col]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"timestamp", "symbol", "interval", "market", "exchange", "ingested_at", "date",
               "label_1h", "label_4h", "label_24h", "label_7d", "open_time", "close_time",
               "open", "high", "low", "close", "volume", "quote_volume", "trades",
               "source_mode", "provider"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _check_provenance(df: pd.DataFrame) -> str:
    """Return a strict provenance label; unknown/mixed data is never treated as real."""
    if df.empty or "source_mode" not in df.columns:
        return "UNKNOWN"
    modes = set(df["source_mode"].dropna().astype(str).unique())
    if not modes:
        return "UNKNOWN"
    real_modes = {SOURCE_LIVE, SOURCE_HISTORICAL}
    if modes == {SOURCE_SYNTHETIC}:
        return SOURCE_SYNTHETIC
    if modes.issubset(real_modes):
        return next(iter(modes)) if len(modes) == 1 else "REAL_MIXED"
    if modes == {SOURCE_SYNTHETIC, SOURCE_LIVE} or modes == {SOURCE_SYNTHETIC, SOURCE_HISTORICAL} or modes == {SOURCE_SYNTHETIC, SOURCE_LIVE, SOURCE_HISTORICAL}:
        return "MIXED_OR_UNKNOWN"
    return "UNKNOWN"


def train_model(df: pd.DataFrame, horizon: str, params: dict = None,
                allow_synthetic: bool = False) -> tuple[Optional[lgb.Booster], dict]:
    """Train with timestamp-based walk-forward folds and a horizon-sized embargo."""
    if params is None:
        params = dict(BASE_PARAMS)

    provenance = _check_provenance(df)
    modes = set(df.get("source_mode", pd.Series(dtype=str)).dropna().astype(str).unique())
    real_modes = {SOURCE_LIVE, SOURCE_HISTORICAL}
    if not allow_synthetic and (not modes or not modes.issubset(real_modes)):
        raise ValueError(f"Refusing production training on non-real, mixed, or unknown provenance: {sorted(modes)}")
    if not allow_synthetic and provenance in {"MIXED_OR_UNKNOWN", "UNKNOWN", SOURCE_SYNTHETIC}:
        raise ValueError(f"Refusing production training for provenance: {provenance}")

    label_col = f"label_{horizon}"
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    df_v = df.copy()
    df_v["timestamp"] = pd.to_datetime(df_v["timestamp"], utc=True, errors="coerce")
    df_v[label_col] = pd.to_numeric(df_v[label_col], errors="coerce")
    df_v = df_v.replace([np.inf, -np.inf], np.nan)
    df_v = df_v.dropna(subset=["timestamp", label_col]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    feats = _feature_cols(df_v)
    if not feats:
        raise ValueError("No numeric feature columns available")
    if len(df_v) < 100:
        raise ValueError(f"Not enough labelled rows for {horizon}: {len(df_v)}")

    params = dict(params, objective="regression", seed=42, verbosity=-1)
    unique_times = pd.Index(df_v["timestamp"].drop_duplicates().sort_values())
    gap_bars = max(int(GAP), int(HORIZON_BARS.get(horizon, GAP)))
    fold_ics = []
    fold_rows = []

    # Expanding-window evaluation by timestamp prevents the same market instant from
    # being split across train/validation. The embargo prevents training labels from
    # reaching into the validation window, especially for 7-day crypto targets.
    for fold in range(1, N_FOLDS):
        val_start_idx = fold * len(unique_times) // N_FOLDS
        val_end_idx = (fold + 1) * len(unique_times) // N_FOLDS
        if val_start_idx >= len(unique_times) or val_end_idx <= val_start_idx:
            continue
        train_end_idx = max(0, val_start_idx - gap_bars)
        if train_end_idx <= 0:
            continue

        train_times = unique_times[:train_end_idx]
        val_times = unique_times[val_start_idx:val_end_idx]
        tr = df_v[df_v["timestamp"].isin(train_times)]
        va = df_v[df_v["timestamp"].isin(val_times)]
        if len(tr) < 100 or len(va) < 30:
            continue

        dtrain = lgb.Dataset(tr[feats], label=tr[label_col])
        dval = lgb.Dataset(va[feats], label=va[label_col], reference=dtrain)
        model = lgb.train(
            params, dtrain, num_boost_round=ROUNDS, valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred_values = model.predict(va[feats])
        scored = va[["timestamp", label_col]].copy()
        scored["prediction"] = pred_values
        per_time_ic = []
        for _, snapshot in scored.groupby("timestamp", sort=True):
            if len(snapshot) < 5 or snapshot["prediction"].nunique() < 2 or snapshot[label_col].nunique() < 2:
                continue
            ic, _ = spearmanr(snapshot["prediction"], snapshot[label_col])
            if np.isfinite(ic):
                per_time_ic.append(float(ic))
        fold_ics.append(float(np.mean(per_time_ic)) if per_time_ic else float("nan"))
        fold_rows.append({"fold": fold, "train_rows": int(len(tr)), "validation_rows": int(len(va)),
                          "validation_start": str(val_times[0]), "validation_end": str(val_times[-1]),
                          "timestamp_ic_count": len(per_time_ic)})

    finite_ics = [ic for ic in fold_ics if np.isfinite(ic)]
    final_model = lgb.train(params, lgb.Dataset(df_v[feats], label=df_v[label_col]), num_boost_round=ROUNDS)

    metrics = {
        "horizon": horizon,
        "n_rows": int(len(df_v)),
        "n_symbols": int(df_v["symbol"].nunique()) if "symbol" in df_v else 0,
        "n_timestamps": int(df_v["timestamp"].nunique()),
        "n_features": len(feats),
        "features": feats[:30],
        "fold_ics": [round(ic, 4) if np.isfinite(ic) else None for ic in fold_ics],
        "mean_fold_ic": round(float(np.mean(finite_ics)), 4) if finite_ics else None,
        "folds": fold_rows,
        "embargo_bars": gap_bars,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "params": {k: v for k, v in params.items() if k in ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction", "lambda_l2", "max_bin", "extra_trees")},
        "data_provenance": provenance,
        "allow_synthetic": allow_synthetic,
        "metric_note": "Mean cross-sectional Spearman IC over validation timestamps; not live performance.",
    }
    return final_model, metrics

def save_model(model: lgb.Booster, horizon: str, provenance: str = "UNKNOWN") -> Path:
    path = MODELS / f"crypto_{horizon}.txt"
    model.save_model(str(path))
    # Save provenance alongside model
    prov_path = MODELS / f"crypto_{horizon}_provenance.json"
    prov_path.write_text(json.dumps({
        "horizon": horizon,
        "provenance": provenance,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(path),
    }, indent=2))
    return path


def run_training(interval: str = "1h", allow_synthetic: bool = False) -> dict:
    """Run full training pipeline. Refuses to train on synthetic-only data unless allow_synthetic=True."""
    log.info("Starting training pipeline on interval %s (allow_synthetic=%s)", interval, allow_synthetic)
    df = build_feature_dataset(interval=interval, require_real=not allow_synthetic)
    if df.empty:
        raise RuntimeError("No feature data available for training")
    
    save_features(df, "training")
    provenance = _check_provenance(df)
    
    results = {}
    for horizon in HORIZONS:
        if horizon == "7d" and interval != "1h":
            continue
        try:
            model, metrics = train_model(df, horizon, allow_synthetic=allow_synthetic)
            if model is not None:
                path = save_model(model, horizon, provenance)
                metrics["model_path"] = str(path)
                results[horizon] = metrics
                log.info("Trained %s: mean IC=%s, n=%d, provenance=%s", horizon, metrics.get("mean_fold_ic"), metrics["n_rows"], provenance)
            else:
                results[horizon] = {"error": "Model is None", "provenance": provenance}
        except Exception as e:
            log.error("Training failed for %s: %s", horizon, e)
            results[horizon] = {"error": str(e), "provenance": provenance}
    
    report_path = REPORTS / "training_report.json"
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "interval": interval,
        "n_rows": len(df),
        "symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
        "data_provenance": provenance,
        "horizons": results,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    
    return report


def predict(df: pd.DataFrame, horizon: str, allow_synthetic: bool = False) -> Optional[pd.Series]:
    """Generate forecasts only when model and input provenance are explicitly permitted."""
    model_path = MODELS / f"crypto_{horizon}.txt"
    if not model_path.exists():
        return None

    provenance_path = MODELS / f"crypto_{horizon}_provenance.json"
    try:
        model_provenance = json.loads(provenance_path.read_text()).get("provenance", "UNKNOWN")
    except (OSError, ValueError):
        model_provenance = "UNKNOWN"

    real_modes = {SOURCE_LIVE, SOURCE_HISTORICAL}
    data_provenance = _check_provenance(df)
    data_modes = set(df.get("source_mode", pd.Series(dtype=str)).dropna().astype(str).unique())
    if not allow_synthetic:
        if not data_modes or not data_modes.issubset(real_modes) or data_provenance in {"UNKNOWN", "MIXED_OR_UNKNOWN", SOURCE_SYNTHETIC}:
            raise ValueError(f"Refusing production prediction for data provenance: {data_provenance}")
        if model_provenance not in real_modes | {"REAL_MIXED"}:
            raise ValueError(f"Refusing production prediction from model provenance: {model_provenance}")

    model = lgb.Booster(model_file=str(model_path))
    expected = list(model.feature_name())
    missing = [name for name in expected if name not in df.columns]
    if missing:
        raise ValueError(f"Prediction features missing from input: {missing[:10]}")
    return pd.Series(model.predict(df[expected]), index=df.index, name=f"pred_{horizon}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    report = run_training(allow_synthetic=False)
    print(json.dumps(report, indent=2, default=str))
