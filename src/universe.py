"""Crypto universe discovery and management."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import DATA, PRIMARY_EXCHANGE, PRIMARY_MARKET, UNIVERSE_SIZE

log = logging.getLogger(__name__)
UNIVERSE_PATH = DATA / "state" / "universe.json"
SNAPSHOT_DIR = DATA / "state" / "universe_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# Synthetic fallback universe for offline/testing
SYNTHETIC_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "DOTUSDT", "MATICUSDT", "AVAXUSDT", "LTCUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT",
    "XLMUSDT", "ALGOUSDT", "VETUSDT", "FILUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "SUIUSDT", "SEIUSDT", "NEARUSDT", "AAVEUSDT", "MKRUSDT", "RUNEUSDT",
    "IMXUSDT", "INJUSDT", "PEPEUSDT", "WIFUSDT", "FLOKIUSDT", "BONKUSDT", "SATSUSDT",
    "RENDERUSDT", "THETAUSDT", "HNTUSDT", "ENAUSDT", "TRXUSDT", "ETCUSDT",
    "XMRUSDT", "DASHUSDT", "ZECUSDT", "KSMUSDT", "RENUSDT", "1INCHUSDT", "ENJUSDT",
    "BATUSDT", "ZRXUSDT", "COMPUSDT", "SNXUSDT", "SUSHIUSDT", "YFIUSDT", "CRVUSDT",
    "BALUSDT", "OMUSDT", "LRCUSDT", "GRTUSDT", "STORJUSDT", "ANTUSDT", "CVXUSDT",
    "FXSUSDT", "GLMUSDT", "TIAUSDT", "PIXELUSDT", "PORTALUSDT", "JUPUSDT", "WLDUSDT",
]


def discover_binance_universe(market_type: str = "spot", quote_asset: str = "USDT", limit: int = 200) -> pd.DataFrame:
    """Discover active symbols from Binance exchange info."""
    base_url = "https://api.binance.com" if market_type == "spot" else "https://fapi.binance.com"
    url = f"{base_url}/api/v3/exchangeInfo" if market_type == "spot" else f"{base_url}/fapi/v1/exchangeInfo"
    
    log.info("Fetching Binance %s exchange info...", market_type)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    info = r.json()
    
    rows = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if not s.get("quoteAsset", "").endswith(quote_asset):
            continue
        if market_type == "spot" and s.get("contractType") and s["contractType"] != "SPOT":
            continue
        if market_type == "futures" and s.get("contractType") not in ("PERPETUAL",):
            continue
        rows.append({
            "symbol": s["symbol"],
            "base_asset": s["baseAsset"],
            "quote_asset": s["quoteAsset"],
            "status": s["status"],
            "contract_type": s.get("contractType", "SPOT"),
            "exchange": "binance",
            "market": market_type,
        })
    
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No %s pairs found with quote %s", market_type, quote_asset)
        return df
    
    log.info("Found %d active %s %s pairs", len(df), market_type, quote_asset)
    return df


def build_universe(max_symbols: int = UNIVERSE_SIZE, include_futures: bool = True) -> pd.DataFrame:
    """Build the full tradable universe. Falls back to synthetic if network is unavailable."""
    try:
        spot = discover_binance_universe("spot", "USDT", limit=max_symbols * 2)
    except Exception as e:
        log.warning("Cannot reach Binance (%s). Using synthetic universe.", e)
        spot = pd.DataFrame()
    
    if spot.empty:
        # Synthetic fallback
        symbols = SYNTHETIC_UNIVERSE[:max_symbols]
        rows = []
        for sym in symbols:
            base = sym.replace("USDT", "")
            rows.append({
                "symbol": sym,
                "base_asset": base,
                "quote_asset": "USDT",
                "status": "TRADING",
                "contract_type": "SPOT",
                "exchange": "binance",
                "market": "spot",
            })
        spot = pd.DataFrame(rows)
        log.info("Using synthetic fallback universe: %d symbols", len(spot))
    
    spot["priority"] = 0
    spot = spot.sort_values(["priority", "base_asset"]).reset_index(drop=True)
    selected = spot.head(max_symbols).copy()
    selected["listed_at"] = datetime.now(timezone.utc).isoformat()
    selected["in_model"] = True
    
    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "binance_spot_rest" if not spot.empty and spot.iloc[0]["exchange"] == "binance" else "synthetic_fallback",
        "count": len(selected),
        "symbols": selected["symbol"].tolist(),
    }
    snap_path = SNAPSHOT_DIR / f"universe_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    snap_path.write_text(json.dumps(snap, indent=2))
    
    UNIVERSE_PATH.write_text(json.dumps({
        "generated_at": snap["generated_at"],
        "symbols": selected.to_dict(orient="records"),
        "count": len(selected),
    }, indent=2))
    
    log.info("Universe saved: %d symbols", len(selected))
    return selected


def load_universe() -> pd.DataFrame:
    """Load the current universe from disk."""
    if not UNIVERSE_PATH.exists():
        return build_universe()
    data = json.loads(UNIVERSE_PATH.read_text())
    return pd.DataFrame(data.get("symbols", []))


def universe_symbols() -> list[str]:
    return load_universe()["symbol"].tolist()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_universe(60)
    print(df[["symbol", "base_asset", "quote_asset"]].to_string(index=False))

