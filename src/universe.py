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

from config import DATA, PRIMARY_EXCHANGE, PRIMARY_MARKET, UNIVERSE_SIZE, SOURCE_SYNTHETIC, SOURCE_HISTORICAL
from http_client import http_get

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
    r = http_get(url, timeout=8)
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


def discover_kraken_universe(quote_asset: str = "USDT", limit: int = 200) -> pd.DataFrame:
    """Discover active exact-quote Kraken spot pairs; never relabel USD data as USDT."""
    response = http_get("https://api.kraken.com/0/public/AssetPairs", timeout=20)
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken AssetPairs error: {payload['error']}")
    rows = []
    for pair_key, item in payload.get("result", {}).items():
        wsname = str(item.get("wsname", ""))
        if "/" not in wsname or item.get("status", "online") != "online":
            continue
        base, quote = (part.upper() for part in wsname.split("/", 1))
        base = "BTC" if base == "XBT" else base
        if quote != quote_asset.upper():
            continue
        preferred_order = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "BCH"]
        priority = preferred_order.index(base) if base in preferred_order else len(preferred_order) + 1
        rows.append({
            "symbol": f"{base}{quote}",
            "base_asset": base,
            "quote_asset": quote,
            "status": "TRADING",
            "contract_type": "SPOT",
            "exchange": "kraken",
            "market": "spot",
            "exchange_pair": item.get("altname", pair_key),
            "universe_source": "KRAKEN_ASSET_PAIRS",
            "universe_provenance": SOURCE_HISTORICAL,
            "priority": priority,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["symbol"]).sort_values(["priority", "base_asset"]).head(limit).reset_index(drop=True)
    return df


def build_universe(max_symbols: int = UNIVERSE_SIZE, include_futures: bool = True,
                   preferred_exchange: Optional[str] = None) -> pd.DataFrame:
    """Build a real exchange-derived universe; synthetic symbols remain demo-only."""
    selected_source = "SYNTHETIC_DEMO"
    spot = pd.DataFrame()

    preferred = str(preferred_exchange or "").lower()
    order = [preferred] if preferred in {"binance", "kraken"} else []
    order += [name for name in ("binance", "kraken") if name not in order]

    for exchange in order:
        if not spot.empty:
            break
        if exchange == "binance":
            try:
                spot = discover_binance_universe("spot", "USDT", limit=max_symbols * 2)
                if not spot.empty:
                    selected_source = "BINANCE_EXCHANGE_INFO"
                    spot["universe_source"] = selected_source
                    spot["universe_provenance"] = SOURCE_HISTORICAL
            except Exception as exc:
                log.warning("Binance universe unavailable: %s", exc)
        elif exchange == "kraken":
            # Prefer USDT instruments where actually listed; otherwise keep the
            # genuine USD quote and preserve that symbol as the canonical instrument.
            for quote in ("USDT", "USD"):
                try:
                    spot = discover_kraken_universe(quote, limit=max_symbols * 2)
                    if not spot.empty:
                        selected_source = f"KRAKEN_ASSET_PAIRS_{quote}"
                        break
                except Exception as exc:
                    log.warning("Kraken %s universe unavailable: %s", quote, exc)

    if spot.empty:
        rows = []
        for sym in SYNTHETIC_UNIVERSE[:max_symbols]:
            base = sym[:-4] if sym.endswith("USDT") else sym
            rows.append({
                "symbol": sym, "base_asset": base, "quote_asset": "USDT",
                "status": "DEMO_ONLY", "contract_type": "SPOT",
                "exchange": "synthetic", "market": "spot",
                "universe_source": "SYNTHETIC_DEMO",
                "universe_provenance": SOURCE_SYNTHETIC,
            })
        spot = pd.DataFrame(rows)
        selected_source = "SYNTHETIC_DEMO"
        log.warning("No exchange metadata reachable; using %d demo-only symbols", len(spot))

    if "priority" not in spot.columns:
        spot["priority"] = 0
    spot = spot.sort_values(["priority", "base_asset"]).reset_index(drop=True)
    selected = spot.head(max_symbols).copy()
    selected["listed_at"] = datetime.now(timezone.utc).isoformat()
    selected["in_model"] = selected["universe_provenance"].isin([SOURCE_HISTORICAL]) if "universe_provenance" in selected else selected["exchange"].ne("synthetic")

    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": selected_source,
        "provenance": SOURCE_SYNTHETIC if selected_source == "SYNTHETIC_DEMO" else SOURCE_HISTORICAL,
        "count": len(selected),
        "symbols": selected["symbol"].tolist(),
    }
    snap_path = SNAPSHOT_DIR / f"universe_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    snap_path.write_text(json.dumps(snap, indent=2))

    UNIVERSE_PATH.write_text(json.dumps({
        "generated_at": snap["generated_at"],
        "source": selected_source,
        "provenance": snap["provenance"],
        "symbols": selected.to_dict(orient="records"),
        "count": len(selected),
    }, indent=2))

    log.info("Universe saved: %d symbols from %s", len(selected), selected_source)
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

