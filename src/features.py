"""Crypto-specific feature engineering with provenance tracking."""
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
FEATURES_DIR = DATA / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _load_real_candles(symbol: str, interval: str, market_type: str) -> pd.DataFrame:
    """Merge only exchange-derived rows, with live updates taking priority over history."""
    from market_data import get_candles
    frames = []
    for mode in (SOURCE_HISTORICAL, SOURCE_LIVE):
        part = get_candles(symbol, interval, market_type, source_mode=mode)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "source_mode" not in df.columns or "provider" not in df.columns:
        return pd.DataFrame()
    df["_source_priority"] = df["source_mode"].map({SOURCE_HISTORICAL: 1, SOURCE_LIVE: 2}).fillna(0)
    df["_ingested_sort"] = pd.to_datetime(df.get("ingested_at"), utc=True, errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "_source_priority", "_ingested_sort"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df.drop(columns=["_source_priority", "_ingested_sort"], errors="ignore").reset_index(drop=True)


def compute_features(symbol: str, interval: str = "1h", market_type: str = "spot", require_real: bool = False) -> Optional[pd.DataFrame]:
    """Compute all features for a symbol from stored OHLCV data.
    
    Args:
        require_real: If True, reject datasets that contain only synthetic data.
    """
    from market_data import get_candles

    # Combine historical and live EXCHANGE data only; live rows replace historical
    # rows for identical timestamps. Synthetic fixtures are a separate opt-in path.
    df = _load_real_candles(symbol, interval, market_type)
    if df.empty and not require_real:
        df = get_candles(symbol, interval, market_type, source_mode=SOURCE_SYNTHETIC)

    if df.empty:
        return None

    allowed_real = {SOURCE_HISTORICAL, SOURCE_LIVE}
    if require_real:
        if "source_mode" not in df.columns or not set(df["source_mode"].dropna().astype(str)).issubset(allowed_real):
            log.warning("Rejecting non-real or unprovenanced data for production features: %s %s", symbol, interval)
            return None
    elif "source_mode" not in df.columns or df["source_mode"].nunique() != 1:
        log.warning("Rejecting ambiguous provenance in demo feature series: %s %s", symbol, interval)
        return None
    
    if df.empty or len(df) < 60:
        return None
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    c = df["close"].astype("float64")
    o, h, l = df["open"].astype("float64"), df["high"].astype("float64"), df["low"].astype("float64")
    v = df["volume"].astype("float64").clip(lower=0)
    lc = np.log(c)
    r1 = lc.diff()
    
    f = pd.DataFrame(index=df.index)
    f["timestamp"] = df["timestamp"]
    f["symbol"] = symbol
    f["interval"] = interval
    f["market"] = market_type
    f["source_mode"] = df["source_mode"].astype(str).to_numpy()
    f["provider"] = df["provider"].astype(str).to_numpy() if "provider" in df.columns else "unknown"
    f["close"] = c
    
    # Returns
    for n in (1, 2, 3, 5, 10, 20, 60, 120, 240, 480):
        if n < len(f):
            f[f"ret_{n}"] = lc.diff(n)
    
    f["mom_12_1"] = f.get("ret_240", pd.Series(np.nan, index=f.index)) - f.get("ret_20", pd.Series(np.nan, index=f.index))
    
    # Volatility
    for n in (10, 20, 60, 120):
        if n < len(f):
            f[f"vol_{n}"] = r1.rolling(n).std() * np.sqrt(365 * 24 if interval == "1h" else 365)
    f["vol_ratio"] = f.get("vol_20", pd.Series(np.nan, index=f.index)) / f.get("vol_60", pd.Series(np.nan, index=f.index)).replace(0, np.nan)
    f["down_vol_60"] = r1.clip(upper=0).rolling(60).std() * np.sqrt(365 * 24 if interval == "1h" else 365)
    
    # RSI
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_5"] = _rsi(c, 5)
    
    # Moving averages
    for n in (10, 20, 50, 100, 200):
        if n < len(f):
            ma = c.rolling(n).mean()
            f[f"dist_ma{n}"] = c / ma - 1
    if 50 < len(f) and 200 < len(f):
        f["ma_cross_50_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    
    # High/low features
    if 252 < len(f):
        f["dist_hi252"] = c / c.rolling(252, min_periods=120).max() - 1
        f["dist_lo252"] = c / c.rolling(252, min_periods=120).min() - 1
    if 63 < len(f):
        f["dist_hi63"] = c / c.rolling(63).max() - 1
        f["dist_lo63"] = c / c.rolling(63).min() - 1
    
    # ATR
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean() / c
    f["atr_14_pct"] = f["atr_14"] * 100
    
    # Volume features
    f["vol_change_1"] = v.diff(1) / v.replace(0, np.nan)
    f["vol_ma20"] = v / v.rolling(20).mean().replace(0, np.nan)
    f["vol_ma60"] = v / v.rolling(60).mean().replace(0, np.nan)
    
    # Range
    f["range_1"] = (h - l) / c
    f["range_5"] = (h.rolling(5).max() - l.rolling(5).min()) / c
    
    # Drawdown
    roll_max = c.expanding().max()
    f["drawdown"] = c / roll_max - 1
    f["max_drawdown_20"] = c.rolling(20).apply(lambda x: (x / x.cummax() - 1).min(), raw=False)
    
    # Labels (future returns) - compute only for completed bars
    for h_name, h_bars in HORIZON_BARS.items():
        if h_bars < len(f):
            f[f"label_{h_name}"] = lc.shift(-h_bars) - lc
    
    # BTC regime feature
    try:
        btc = _load_real_candles("BTCUSDT", interval, market_type)
        if btc.empty and not require_real:
            btc = get_candles("BTCUSDT", interval, market_type, source_mode=SOURCE_SYNTHETIC)
        if not btc.empty:
            btc = btc.sort_values("timestamp").reset_index(drop=True)
            btc_lc = np.log(btc["close"].astype("float64"))
            btc_ret_24 = btc_lc.diff(24)
            merged = pd.merge(f, pd.DataFrame({"timestamp": df["timestamp"], "btc_ret_24": btc_ret_24.values}), on="timestamp", how="left")
            f["btc_ret_24"] = merged["btc_ret_24"].values
            if len(f) > 60:
                f["beta_btc_60"] = r1.rolling(60).corr(btc_lc.diff().reindex(f.index, method="nearest"))
    except Exception:
        pass
    
    # Drop rows with insufficient history
    f = f.dropna(subset=["ret_1", "ret_20"]).reset_index(drop=True)
    
    return f


def build_feature_dataset(symbols: list[str] = None, interval: str = "1h", market_type: str = "spot", require_real: bool = False) -> pd.DataFrame:
    """Build a feature dataset for multiple symbols.
    
    Args:
        require_real: If True, reject any symbol whose data is synthetic-only.
    """
    from universe import load_universe
    if symbols is None:
        df = load_universe()
        symbols = df["symbol"].tolist()[:60] if not df.empty else ["BTCUSDT", "ETHUSDT"]
    
    all_feats = []
    rejected = []
    for sym in symbols:
        try:
            feat = compute_features(sym, interval, market_type, require_real=require_real)
            if feat is not None and not feat.empty:
                all_feats.append(feat)
            else:
                rejected.append(sym)
        except Exception as e:
            log.warning("Feature error for %s: %s", sym, e)
            rejected.append(sym)
    
    if rejected:
        log.info("Rejected %d symbols (no real data): %s", len(rejected), rejected[:10])
    
    if not all_feats:
        return pd.DataFrame()
    
    combined = pd.concat(all_feats, ignore_index=True)
    
    # Add market-relative ranks
    combined["date"] = pd.to_datetime(combined["timestamp"]).dt.floor("D")
    for col in ["ret_1", "ret_20", "vol_20", "vol_ratio", "rsi_14"]:
        if col in combined.columns:
            combined[f"{col}_rank"] = combined.groupby(["date", "interval", "market"])[col].rank(pct=True)
    
    return combined


def save_features(df: pd.DataFrame, name: str = "latest"):
    """Save feature dataset to Parquet."""
    path = FEATURES_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    log.info("Saved %d rows to %s (source: %s)", len(df), path, df["source_mode"].iloc[0] if "source_mode" in df.columns and not df.empty else "unknown")


def load_features(name: str = "latest") -> pd.DataFrame:
    path = FEATURES_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    df = build_feature_dataset(["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"], "1h", require_real=False)
    print(f"Features: {len(df)} rows, {len(df.columns)} columns")
    print(df[["timestamp", "symbol", "ret_1", "ret_20", "rsi_14", "vol_ratio", "label_1h", "label_4h", "label_24h", "source_mode"]].tail(5).to_string(index=False))
