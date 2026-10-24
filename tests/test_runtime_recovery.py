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
