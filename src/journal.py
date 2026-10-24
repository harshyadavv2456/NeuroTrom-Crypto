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
                      source_mode: str = SOURCE_SYNTHETIC, data_quality: str = "ok",
                      data_cutoff_timestamp: Optional[datetime] = None) -> dict:
    """Record a forecast once per symbol/horizon/model/input-candle cutoff."""
    if source_mode == SOURCE_SYNTHETIC:
        log.warning("Recording SYNTHETIC prediction for %s %s — not a real market forecast", symbol, horizon)
    
    created = pd.Timestamp(timestamp)
    created = created.tz_localize("UTC") if created.tzinfo is None else created.tz_convert("UTC")
    cutoff = pd.Timestamp(data_cutoff_timestamp or timestamp)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    prediction_id = f"{symbol}_{horizon}_{model_version}_{cutoff.isoformat()}"
    pred = {
        "prediction_id": prediction_id,
        "symbol": symbol,
        "horizon": horizon,
        "timestamp": created.isoformat(),
        "data_cutoff_timestamp": cutoff.isoformat(),
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
            existing = pd.DataFrame()

    if not existing.empty and "prediction_id" in existing.columns:
        already = existing[existing["prediction_id"].astype(str) == prediction_id]
        if not already.empty:
            return already.iloc[-1].to_dict()

    new_row = pd.DataFrame([pred])
    combined = pd.concat([existing, new_row], ignore_index=True)
    tmp = PREDICTIONS.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False, engine="pyarrow")
    tmp.replace(PREDICTIONS)

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
        cutoff_value = row.get("data_cutoff_timestamp", row["timestamp"])
        cutoff = pd.Timestamp(cutoff_value)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        entry_price = float(row["price_at_prediction"])
        bars = int(HORIZON_BARS.get(horizon, 24))

        df = get_candles(sym, "1h", "spot", start=cutoff + pd.Timedelta(hours=1),
                         end=cutoff + pd.Timedelta(hours=bars + 3))
        if df.empty or "source_mode" not in df.columns:
            continue
        df = df[df["source_mode"].isin([SOURCE_LIVE, SOURCE_HISTORICAL])].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        if len(df) < bars:
            continue

        exit_price = float(df.iloc[bars - 1]["close"])
        actual_ret = (exit_price / entry_price) - 1
        
        idx = preds.index[preds["prediction_id"].astype(str) == str(row["prediction_id"])]
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
