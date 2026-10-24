from types import SimpleNamespace

import pandas as pd
import pytest
import requests

import live
import market_data
from config import SOURCE_HISTORICAL


def _universe(symbol: str, exchange: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": symbol,
        "exchange": exchange,
        "universe_provenance": SOURCE_HISTORICAL,
        "status": "ACTIVE",
    }])


def test_universe_is_preserved_when_all_fast_probes_fail(monkeypatch):
    known = _universe("BTCUSDT", "kraken")
    monkeypatch.setattr(live, "load_universe", lambda: known.copy())

    def unexpected_build(*args, **kwargs):
        pytest.fail("universe must not be rebuilt without any reachable probe")

    monkeypatch.setattr(live, "build_universe", unexpected_build)
    result = live._refresh_universe_if_needed({
        "binance": {"reachable": False, "error": "timeout"},
        "kraken": {"reachable": False, "error": "connection reset"},
    })
    assert result["symbol"].tolist() == ["BTCUSDT"]
    assert result["exchange"].tolist() == ["kraken"]


def test_universe_recovery_uses_provider_status_argument(monkeypatch):
    monkeypatch.setattr(live, "load_universe", lambda: _universe("BTCUSDT", "binance"))
    calls = {}

    def fake_build(size, preferred_exchange):
        calls["size"] = size
        calls["preferred_exchange"] = preferred_exchange
        return _universe("ETHUSDT", "kraken")

    monkeypatch.setattr(live, "build_universe", fake_build)
    result = live._refresh_universe_if_needed({
        "binance": {"reachable": False},
        "kraken": {"reachable": True},
    })
    assert calls["preferred_exchange"] == "kraken"
    assert result["symbol"].tolist() == ["ETHUSDT"]


def test_binance_rest_fails_over_to_next_official_host(monkeypatch):
    calls = []

    def fake_http_get(url, params=None, timeout=15):
        calls.append(url)
        if "api.binance.com" in url and "api-gcp.binance.com" not in url:
            raise requests.ConnectionError("network blocked")
        return SimpleNamespace(status_code=200, url=url)

    monkeypatch.setattr(market_data, "http_get", fake_http_get)
    response = market_data._binance_get("/api/v3/ping", timeout=1)
    assert response.status_code == 200
    assert "api-gcp.binance.com" in response.url
    assert len(calls) == 2


def test_binance_connectivity_probe_checks_alternate_hosts(monkeypatch):
    def fake_http_get(url, params=None, timeout=15):
        if "api.binance.com" in url and "api-gcp.binance.com" not in url:
            raise requests.ConnectionError("network blocked")
        return SimpleNamespace(
            status_code=200,
            headers={"X-NeuroTrom-Transport": "curl-fallback"},
        )

    monkeypatch.setattr(market_data, "http_get", fake_http_get)
    result = market_data.test_provider_connectivity("binance")
    assert result["reachable"] is True
    assert result["host"] == "https://api-gcp.binance.com"
    assert result["transport"] == "curl-fallback"



def test_coindcx_market_discovery_preserves_exact_quote(monkeypatch):
    markets = [
        {"status": "active", "symbol": "BTCUSDT", "target_currency_short_name": "BTC",
         "base_currency_short_name": "USDT", "pair": "B-BTC_USDT"},
        {"status": "active", "symbol": "BTCINR", "target_currency_short_name": "BTC",
         "base_currency_short_name": "INR", "pair": "I-BTC_INR"},
        {"status": "inactive", "symbol": "ETHUSDT", "target_currency_short_name": "ETH",
         "base_currency_short_name": "USDT", "pair": "B-ETH_USDT"},
    ]
    monkeypatch.setattr("universe.http_get", lambda *a, **k: SimpleNamespace(json=lambda: markets))
    result = __import__("universe").discover_coindcx_universe("USDT")
    assert result["symbol"].tolist() == ["BTCUSDT"]
    assert result.iloc[0]["exchange_pair"] == "B-BTC_USDT"
    assert result.iloc[0]["exchange"] == "coindcx"


def test_coindcx_live_ticker_uses_public_endpoint_only(monkeypatch):
    market_data._COINDCX_TICKERS_CACHE = {}
    market_data._COINDCX_TICKERS_CACHE_AT = 0.0
    calls = []

    def fake_http_get(url, params=None, timeout=15):
        calls.append(url)
        return SimpleNamespace(
            json=lambda: [{
                "market": "BTCUSDT", "last_price": "68000", "change_24_hour": "2.5",
                "volume": "1200", "high": "69000", "low": "66000", "timestamp": 1792860000000,
            }],
            headers={"X-NeuroTrom-Transport": "requests"},
        )

    monkeypatch.setattr(market_data, "http_get", fake_http_get)
    result = market_data.fetch_ticker("BTCUSDT", provider="coindcx")
    assert result["provider"] == "coindcx"
    assert result["price"] == 68000.0
    assert result["source_mode"] == "LIVE_EXCHANGE"
    assert calls == ["https://api.coindcx.com/exchange/ticker"]
    assert not any("/orders/" in url for url in calls)


def test_coindcx_candles_are_normalized_and_unfinished_bar_removed(monkeypatch):
    market_data._COINDCX_MARKETS_CACHE = [{
        "status": "active", "symbol": "BTCUSDT", "coindcx_name": "BTCUSDT", "pair": "B-BTC_USDT"
    }]
    market_data._COINDCX_MARKETS_CACHE_AT = 9999999999
    future_open = int((pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)).timestamp() * 1000)
    old_open = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)).floor("h").timestamp() * 1000)

    def fake_http_get(url, params=None, timeout=15):
        return SimpleNamespace(json=lambda: [
            {"time": future_open, "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"time": old_open, "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
        ])

    monkeypatch.setattr(market_data, "http_get", fake_http_get)
    result = market_data.coindcx_ohlc("BTCUSDT", interval="1h")
    assert len(result) == 1
    assert result.iloc[0]["exchange"] == "coindcx"
    assert result.iloc[0]["source_mode"] == "HISTORICAL_EXCHANGE"
