"""Regression tests for the local real-data pipeline and provider fallback."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest


def test_http_client_uses_verified_curl_fallback(monkeypatch):
    import http_client

    def fail_requests(*args, **kwargs):
        raise http_client.requests.exceptions.SSLError("simulated Python TLS reset")

    monkeypatch.setattr(http_client.requests, "get", fail_requests)
    monkeypatch.setattr(http_client.shutil, "which", lambda name: "/usr/bin/curl")
    payload = '{"result":{"unixtime":123}}'
    fake = SimpleNamespace(
        stdout=payload + http_client._MARKER + "200",
        stderr="",
        returncode=0,
    )
    monkeypatch.setattr(http_client.subprocess, "run", lambda *a, **kw: fake)

    response = http_client.http_get("https://api.example.invalid/0/public/Time", timeout=2)
    assert response.status_code == 200
    assert response.headers["X-NeuroTrom-Transport"] == "curl-fallback"
    assert response.json()["result"]["unixtime"] == 123


def test_candle_range_query_accepts_timezone_aware_datetime(monkeypatch, tmp_path):
    import market_data

    monkeypatch.setattr(market_data, "MARKET", tmp_path)
    target = tmp_path / "spot" / "BTCUSD"
    target.mkdir(parents=True)
    stamps = pd.date_range("2026-10-24T00:00:00Z", periods=4, freq="h")
    pd.DataFrame({
        "timestamp": stamps,
        "open": [100.0] * 4,
        "high": [101.0] * 4,
        "low": [99.0] * 4,
        "close": [100.5] * 4,
        "volume": [1.0] * 4,
        "symbol": ["BTCUSD"] * 4,
        "interval": ["1h"] * 4,
        "market": ["spot"] * 4,
        "exchange": ["kraken"] * 4,
        "ingested_at": [datetime.now(timezone.utc).isoformat()] * 4,
        "source_mode": ["HISTORICAL_EXCHANGE"] * 4,
        "provider": ["kraken"] * 4,
    }).to_parquet(target / "1h.parquet", index=False)

    result = market_data.get_candles(
        "BTCUSD", "1h", start=datetime(2026, 10, 24, 1, tzinfo=timezone.utc),
        end=datetime(2026, 10, 24, 2, tzinfo=timezone.utc),
        source_mode="HISTORICAL_EXCHANGE",
    )
    assert len(result) == 2
    assert str(result.iloc[0]["timestamp"])[:13] == "2026-10-24 01"


def test_kraken_symbol_resolver_requires_exact_quote(monkeypatch):
    import market_data

    monkeypatch.setattr(market_data, "kraken_asset_pairs", lambda refresh=False: [
        {"pair_key": "XXBTZUSD", "altname": "XBTUSD", "wsname": "XBT/USD", "status": "online"},
        {"pair_key": "XBTUSDT", "altname": "XBTUSDT", "wsname": "XBT/USDT", "status": "online"},
    ])
    assert market_data.resolve_kraken_pair("BTCUSD") == "XBTUSD"
    assert market_data.resolve_kraken_pair("BTCUSDT") == "XBTUSDT"


def test_real_feature_provenance_rejects_demo_mixture():
    from train import _check_provenance
    from config import SOURCE_HISTORICAL, SOURCE_LIVE, SOURCE_SYNTHETIC

    assert _check_provenance(pd.DataFrame({"source_mode": [SOURCE_HISTORICAL] * 3})) == SOURCE_HISTORICAL
    assert _check_provenance(pd.DataFrame({"source_mode": [SOURCE_HISTORICAL, SOURCE_LIVE]})) == "REAL_MIXED"
    assert _check_provenance(pd.DataFrame({"source_mode": [SOURCE_HISTORICAL, SOURCE_SYNTHETIC]})) == "MIXED_OR_UNKNOWN"
    assert _check_provenance(pd.DataFrame({"other": [1, 2, 3]})) == "UNKNOWN"


def test_horizons_use_crypto_24_7_hours():
    from config import HORIZON_BARS
    assert HORIZON_BARS == {"1h": 1, "4h": 4, "24h": 24, "7d": 168}
