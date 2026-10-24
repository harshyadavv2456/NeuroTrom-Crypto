from __future__ import annotations

import pandas as pd

from scripts import cloud_pipeline as cp


def test_binance_host_probe_falls_back(monkeypatch):
    calls = []

    def fake_request(url, params=None, timeout=8):
        calls.append(url)
        if "data-api.binance.vision" in url:
            raise OSError("edge unavailable")
        if "api.binance.com/api/v3/ping" in url:
            raise OSError("primary unavailable")
        return {}

    monkeypatch.setattr(cp, "_request_json", fake_request)
    host, checks = cp.choose_binance_host()
    assert host == "https://api-gcp.binance.com"
    assert len(checks) == 3
    assert len(calls) == 3


def test_kaggle_parser_accepts_exact_hourly_usdt_pairs(tmp_path):
    stamp = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": stamp.astype(str),
        "symbol": ["BTCUSDT"] * len(stamp),
        "open": range(100, 100 + len(stamp)),
        "high": range(101, 101 + len(stamp)),
        "low": range(99, 99 + len(stamp)),
        "close": range(100, 100 + len(stamp)),
        "volume": [10.0] * len(stamp),
    })
    frame.to_csv(tmp_path / "hourly.csv", index=False)
    result = cp.parse_compatible_kaggle_csvs(tmp_path)
    assert list(result) == ["BTCUSDT"]
    assert len(result["BTCUSDT"]) == 240
    assert str(result["BTCUSDT"]["timestamp"].dt.tz) == "UTC"
    assert result["BTCUSDT"]["source"].str.startswith("KAGGLE:").all()


def test_kaggle_parser_rejects_daily_data_instead_of_faking_hourly_bars(tmp_path):
    stamp = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": stamp.astype(str),
        "symbol": ["BTCUSDT"] * len(stamp),
        "open": [100.0] * len(stamp),
        "high": [102.0] * len(stamp),
        "low": [98.0] * len(stamp),
        "close": [101.0] * len(stamp),
        "volume": [10.0] * len(stamp),
    })
    frame.to_csv(tmp_path / "daily.csv", index=False)
    assert cp.parse_compatible_kaggle_csvs(tmp_path) == {}


def test_feature_generation_returns_only_lagged_market_features():
    stamp = pd.date_range("2026-01-01", periods=250, freq="h", tz="UTC")
    close = pd.Series([100 * (1.0005 ** i) for i in range(len(stamp))])
    frame = pd.DataFrame({
        "timestamp": stamp, "open": close * 0.999, "high": close * 1.002,
        "low": close * 0.998, "close": close, "volume": [50.0] * len(stamp),
        "symbol": ["BTCUSDT"] * len(stamp),
    })
    result = cp.add_features(frame)
    assert "ret_1" in result
    assert "volatility_24" in result
    assert "label_7d" not in result
    assert pd.isna(result.iloc[0]["ret_1"])
    assert result["ret_1"].iloc[-1] > 0


def test_forecast_journal_matures_only_after_target_candle_exists():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        "close": [100.0, 110.0, 108.0, 112.0],
        "symbol": ["BTCUSDT"] * 4,
        "source": ["BINANCE_PUBLIC_REST"] * 4,
    })
    record = {
        "journal_id": "run-1:BTCUSDT:1h",
        "symbol": "BTCUSDT",
        "horizon": "1h",
        "input_candle_time": "2026-01-01T00:00:00+00:00",
        "entry_price": 100.0,
        "last_price": 100.0,
        "predicted_return_pct": 5.0,
        "matured": False,
    }
    journal, summary = cp.update_journal([record], [], {"BTCUSDT": frame})
    assert journal[0]["matured"] is True
    assert journal[0]["actual_return_pct"] == 10.0
    assert journal[0]["directional_correct"] is True
    assert summary["matured_predictions"] == 1
    assert summary["directional_accuracy"] == 1.0
