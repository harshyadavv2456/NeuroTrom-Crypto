"""LightGBM training with walk-forward validation."""
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

from config import DATA, HORIZONS, BASE_PARAMS, ROUNDS, N_FOLDS, GAP, COST_BPS
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
               "open", "high", "low", "close", "volume", "quote_volume", "trades"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def train_model(df: pd.DataFrame, horizon: str, params: dict = None) -> tuple[lgb.Booster, dict]:
    """Train a LightGBM model with walk-forward validation."""
    if params is None:
        params = dict(BASE_PARAMS)
    
    params = dict(params, objective="regression", seed=42)
    feats = _feature_cols(df)
    label = _labels(df, horizon)
    
    # Drop rows with missing labels
    valid = label.notna()
    df_v = df[valid].copy()
    label_v = label[valid]
    
    if len(df_v) < 100:
        raise ValueError(f"Not enough data for {horizon}: {len(df_v)} rows")
    
    # Time-based split for walk-forward
    df_v = df_v.sort_values("timestamp").reset_index(drop=True)
    n = len(df_v)
    fold_size = n // N_FOLDS
    gap_rows = max(1, int(GAP))
    
    fold_ics = []
    models = []
    
    for fold in range(N_FOLDS):
        if fold >= N_FOLDS - 1:
            # Last fold: use as training for final model
            break
        
        train_end = fold * fold_size
        val_start = train_end + gap_rows
        val_end = (fold + 1) * fold_size
        
        if val_start >= n or val_end > n:
            continue
        
        tr = df_v.iloc[:train_end]
        va = df_v.iloc[val_start:val_end]
        
        if len(tr) < 100 or len(va) < 30:
            continue
        
        dtrain = lgb.Dataset(tr[feats], label=label_v.iloc[:train_end])
        dval = lgb.Dataset(va[feats], label=label_v.iloc[val_start:val_end], reference=dtrain)
        
        m = lgb.train(params, dtrain, num_boost_round=ROUNDS,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        pred = m.predict(va[feats])
        ic, _ = spearmanr(pred, label_v.iloc[val_start:val_end])
        fold_ics.append(float(ic))
        models.append(m)
    
    # Train final model on all data
    dtrain_all = lgb.Dataset(df_v[feats], label=label_v)
    final_model = lgb.train(params, dtrain_all, num_boost_round=ROUNDS)
    
    metrics = {
        "horizon": horizon,
        "n_rows": int(len(df_v)),
        "n_features": len(feats),
        "features": feats[:20],  # store top 20 for reporting
        "fold_ics": [round(ic, 4) for ic in fold_ics],
        "mean_fold_ic": round(float(np.mean(fold_ics)), 4) if fold_ics else None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "params": {k: v for k, v in params.items() if k in ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction", "lambda_l2", "max_bin", "extra_trees")},
    }
    
    return final_model, metrics


def save_model(model: lgb.Booster, horizon: str) -> Path:
    path = MODELS / f"crypto_{horizon}.txt"
    model.save_model(str(path))
    return path


def run_training(interval: str = "1h") -> dict:
    """Run full training pipeline."""
    log.info("Starting training pipeline on interval %s", interval)
    df = build_feature_dataset(interval=interval)
    if df.empty:
        raise RuntimeError("No feature data available for training")
    
    save_features(df, "training")
    
    results = {}
    for horizon in HORIZONS:
        if horizon == "7d" and interval != "1h":
            continue
        try:
            model, metrics = train_model(df, horizon)
            path = save_model(model, horizon)
            metrics["model_path"] = str(path)
            results[horizon] = metrics
            log.info("Trained %s: mean IC=%s, n=%d", horizon, metrics.get("mean_fold_ic"), metrics["n_rows"])
        except Exception as e:
            log.error("Training failed for %s: %s", horizon, e)
            results[horizon] = {"error": str(e)}
    
    # Save report
    report_path = REPORTS / "training_report.json"
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "interval": interval,
        "n_rows": len(df),
        "symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
        "horizons": results,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    
    return report


def predict(df: pd.DataFrame, horizon: str) -> Optional[pd.Series]:
    """Generate predictions using saved model."""
    from market_data import get_candles
    model_path = MODELS / f"crypto_{horizon}.txt"
    if not model_path.exists():
        return None
    
    model = lgb.Booster(model_file=str(model_path))
    feats = _feature_cols(df)
    valid_feats = [f for f in feats if f in df.columns]
    if not valid_feats:
        return None
    
    return pd.Series(model.predict(df[valid_feats]), index=df.index, name=f"pred_{horizon}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    report = run_training()
    print(json.dumps(report, indent=2, default=str))

