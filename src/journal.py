"""Prediction journal and paper-performance tracking with provenance."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import DATA, HORIZONS, HORIZON_BARS, SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC

log = logging.getLogger(__name__)
JOURNAL = DATA / "journal"
JOURNAL.mkdir(parents=True, exist_ok=True)
PREDICTIONS = JOURNAL / "predictions.parquet"
OUTCOMES = JOURNAL / "outcomes.parquet"
SUMMARY = JOURNAL / "summary.json"


def record_prediction(symbol: str, horizon: str, timestamp: datetime, predicted_return: float,
                      model_version: str, feature_version: str, price: float,
                      source_mode: str = SOURCE_SYNTHETIC, data_quality: str = "ok") -> dict:
    """Record a new prediction before its outcome is known."""
    if source_mode == SOURCE_SYNTHETIC:
        log.warning("Recording SYNTHETIC prediction for %s %s — not a real market forecast", symbol, horizon)
    
    pred = {
        "prediction_id": f"{symbol}_{horizon}_{timestamp.isoformat()}",
        "symbol": symbol,
        "horizon": horizon,
        "timestamp": timestamp.isoformat(),
        "predicted_return": float(predicted_return),
        "model_version": model_version,
        "feature_version": feature_version,
        "price_at_prediction": float(price),
        "source_mode": source_mode,
        "data_quality": data_quality,
        "matured": False,
        "outcome": None,
        "actual_return": None,
        "maturation_time": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    existing = pd.DataFrame()
    if PREDICTIONS.exists():
        try:
            existing = pd.read_parquet(PREDICTIONS)
        except Exception:
            pass
    
    new_row = pd.DataFrame([pred])
    combined = pd.concat([existing, new_row], ignore_index=True)
    combined.to_parquet(PREDICTIONS, index=False, engine="pyarrow")
    
    return pred


def mature_predictions(horizon: str = "24h"):
    """Check which predictions have matured and fill in outcomes."""
    if not PREDICTIONS.exists():
        return 0
    
    preds = pd.read_parquet(PREDICTIONS)
    unmatured = preds[(preds["horizon"] == horizon) & (~preds["matured"])].copy()
    if unmatured.empty:
        return 0
    
    from market_data import get_candles
    matured = 0
    outcomes = []
    
    for _, row in unmatured.iterrows():
        sym = row["symbol"]
        pred_ts = pd.Timestamp(row["timestamp"])
        pred_ts = pred_ts.tz_localize("UTC") if pred_ts.tzinfo is None else pred_ts.tz_convert("UTC")
        entry_price = row["price_at_prediction"]
        bars = int(HORIZON_BARS.get(horizon, 24))

        df = get_candles(sym, "1h", "spot", start=pred_ts, end=pred_ts + pd.Timedelta(hours=bars * 2))
        if df.empty or "source_mode" not in df.columns:
            continue
        # Only exchange-derived bars may score a real prediction; synthetic and
        # unprovenanced records must never silently complete a live forecast.
        df = df[df["source_mode"].isin([SOURCE_LIVE, SOURCE_HISTORICAL])].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            continue
        # Anchor the evaluation on the input candle timestamp saved with the prediction.
        df = df[df["timestamp"] >= pred_ts].reset_index(drop=True)
        if len(df) <= bars:
            continue
        
        exit_price = df.iloc[bars]["close"]
        actual_ret = (exit_price / entry_price) - 1
        
        idx = preds.index[(preds["symbol"] == sym) & (preds["horizon"] == horizon) & (preds["timestamp"] == row["timestamp"])]
        if not idx.empty:
            preds.loc[idx, "matured"] = True
            preds.loc[idx, "outcome"] = "filled"
            preds.loc[idx, "actual_return"] = float(actual_ret)
            preds.loc[idx, "maturation_time"] = datetime.now(timezone.utc).isoformat()
            matured += 1
        
        outcomes.append({
            "symbol": sym,
            "horizon": horizon,
            "timestamp": row["timestamp"],
            "predicted_return": row["predicted_return"],
            "actual_return": float(actual_ret),
            "model_version": row["model_version"],
            "source_mode": row.get("source_mode", SOURCE_SYNTHETIC),
        })
    
    preds.to_parquet(PREDICTIONS, index=False, engine="pyarrow")
    
    if outcomes:
        out_df = pd.DataFrame(outcomes)
        if OUTCOMES.exists():
            old = pd.read_parquet(OUTCOMES)
            out_df = pd.concat([old, out_df], ignore_index=True)
        out_df.to_parquet(OUTCOMES, index=False, engine="pyarrow")
    
    return matured


def compute_journal_summary() -> dict:
    """Compute accuracy and calibration from matured predictions."""
    if not PREDICTIONS.exists():
        return {"note": "no predictions yet", "real_predictions": 0, "synthetic_predictions": 0}
    
    preds = pd.read_parquet(PREDICTIONS)
    total = len(preds)
    modes = preds["source_mode"].astype(str) if "source_mode" in preds.columns else pd.Series(["UNKNOWN"] * len(preds), index=preds.index)
    real_mask = modes.isin([SOURCE_LIVE, SOURCE_HISTORICAL])
    synthetic_mask = modes.eq(SOURCE_SYNTHETIC)
    real = int(real_mask.sum())
    synthetic = int(synthetic_mask.sum())
    unknown = int(total - real - synthetic)
    
    matured = preds[preds["matured"] == True].copy()
    
    summary = {
        "total_predictions": total,
        "real_predictions": real,
        "synthetic_predictions": synthetic,
        "unknown_provenance_predictions": unknown,
        "matured": len(matured),
        "by_horizon": {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    for h in HORIZONS:
        h_preds = matured[matured["horizon"] == h]
        if h_preds.empty:
            summary["by_horizon"][h] = {"n": 0}
            continue
        
        valid = h_preds.dropna(subset=["actual_return", "predicted_return"]).copy()
        actuals = valid["actual_return"]
        predicted = valid["predicted_return"]

        if len(valid) > 10:
            ic, _ = spearmanr(predicted, actuals)
            direction_acc = ((predicted > 0) == (actuals > 0)).mean()
            summary["by_horizon"][h] = {
                "n": int(len(valid)),
                "rank_ic": round(float(ic), 4) if np.isfinite(ic) else None,
                "direction_accuracy": round(float(direction_acc), 4),
                "mean_predicted": round(float(predicted.mean()), 4),
                "mean_actual": round(float(actuals.mean()), 4),
            }
        else:
            summary["by_horizon"][h] = {"n": int(len(valid)), "note": "insufficient matured predictions"}
    
    SUMMARY.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    print(compute_journal_summary())
