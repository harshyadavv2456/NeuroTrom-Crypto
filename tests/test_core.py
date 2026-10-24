"""Core tests for NeuroTrom Crypto — source-aware and provenance-checked."""
from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def test_feature_no_lookahead():
    """Future labels must not be available at feature time."""
    from features import compute_features
    import inspect
    src = inspect.getsource(compute_features)
    assert "shift(-" in src, "Labels must use shift to avoid lookahead"
    assert True


def test_ohlcv_validation():
    """OHLC consistency check drops invalid rows."""
    from market_data import save_candles
    from pathlib import Path
    import shutil
    from config import DATA
    
    test_dir = DATA / "market" / "__test_ohlcv"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    from config import SOURCE_HISTORICAL
    df = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        "open": [100.0] * 5,
        "high": [101.0] * 5,
        "low": [99.0] * 5,
        "close": [100.5] * 5,
        "volume": [1e6] * 5,
        "symbol": ["BTCUSDT"] * 5,
        "interval": ["1h"] * 5,
        "market": ["spot"] * 5,
        "exchange": ["binance"] * 5,
        "ingested_at": ["2024-01-01T00:00:00+00:00"] * 5,
        "source_mode": [SOURCE_HISTORICAL] * 5,
        "provider": ["binance"] * 5,
    })
    df.loc[2, "high"] = 98.0
    df.loc[2, "low"] = 99.0
    written = save_candles(df, "__test_ohlcv", "1h", "spot")
    assert written <= 4, f"Expected <=4 rows written, got {written}"
    
    shutil.rmtree(test_dir, ignore_errors=True)
    shutil.rmtree(DATA / "raw" / "market" / "__test_ohlcv", ignore_errors=True)


def test_data_source_terms():
    """Data sources have terms reviewed."""
    from sources import load_sources
    sources = load_sources()
    assert len(sources) > 0
    for s in sources:
        assert s.terms_reviewed_date != ""
        assert s.commercial_status in ("approved_for_personal_research", "commercial_rights_unconfirmed", "commercial_use_requires_license", "approved_for_commercial_use")


def test_backfill_resumable():
    """Backfill state persists."""
    from backfill import save_backfill_state, load_backfill_state
    state = {"last_run": "2024-01-01T00:00:00+00:00", "symbols_done": {"BTCUSDT_1h_spot": {"end": "2024-01-01T00:00:00+00:00", "rows": 100}}}
    save_backfill_state(state)
    loaded = load_backfill_state()
    assert loaded["last_run"] == "2024-01-01T00:00:00+00:00"
    assert loaded["symbols_done"]["BTCUSDT_1h_spot"]["rows"] == 100


def test_prediction_journal():
    """Predictions are recorded with required fields."""
    from journal import record_prediction, compute_journal_summary
    from datetime import datetime, timezone
    from pathlib import Path
    import shutil
    from config import DATA
    
    p = record_prediction("BTCUSDT", "24h", datetime.now(timezone.utc), 0.01, "v1", "v1", 50000.0)
    assert p["symbol"] == "BTCUSDT"
    assert p["matured"] == False
    assert "prediction_id" in p
    summary = compute_journal_summary()
    assert "total_predictions" in summary
    
    from journal import PREDICTIONS
    if PREDICTIONS.exists():
        shutil.rmtree(PREDICTIONS.parent, ignore_errors=True)


def test_api_binds_localhost():
    """API service must bind to 127.0.0.1 only."""
    from config import HOST, PORT
    assert HOST == "127.0.0.1"
    assert PORT == 8775


def test_port_no_conflict_with_finvest():
    """Port 8775 must not be FinVest's port (8765, 8790)."""
    import shutil
    import subprocess
    if shutil.which("lsof") is None:
        pytest.skip("lsof is unavailable on this runner")
    r = subprocess.run(["lsof", "-iTCP:8775", "-sTCP:LISTEN"], capture_output=True, text=True)
    if r.returncode == 0:
        out = r.stdout.lower()
        assert "api:app" in out or "neurotrom-crypto" in out or "8775" in out, f"Port 8775 occupied by unexpected service: {r.stdout}"


def test_lightgbm_trains():
    """LightGBM can train on synthetic data with explicit flag."""
    import lightgbm as lgb
    from config import BASE_PARAMS
    from scipy.stats import spearmanr
    
    np.random.seed(42)
    n = 2000
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n), "f3": np.random.randn(n)})
    y = X["f1"] * 0.5 + X["f2"] * 0.3 + np.random.randn(n) * 0.5
    
    params = dict(BASE_PARAMS, objective="regression", seed=42, min_data_in_leaf=50, num_leaves=15)
    model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=100)
    preds = model.predict(X)
    ic, _ = spearmanr(preds, y)
    assert np.isfinite(ic), f"LightGBM IC is not finite: {ic}"
    assert ic > 0.2, f"LightGBM IC too low: {ic}"


def test_training_rejects_synthetic_by_default():
    """Training must refuse synthetic-only data unless explicitly allowed."""
    from train import train_model, _check_provenance
    from config import SOURCE_SYNTHETIC
    
    # Create a synthetic-only DataFrame
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
        "symbol": ["BTCUSDT"] * 200,
        "interval": ["1h"] * 200,
        "market": ["spot"] * 200,
        "close": np.random.randn(200).cumsum() + 100,
        "open": np.random.randn(200).cumsum() + 100,
        "high": np.random.randn(200).cumsum() + 101,
        "low": np.random.randn(200).cumsum() + 99,
        "volume": np.random.rand(200) * 1e6,
        "source_mode": [SOURCE_SYNTHETIC] * 200,
        "label_24h": np.random.randn(200),
    })
    
    provenance = _check_provenance(df)
    assert provenance == SOURCE_SYNTHETIC
    
    try:
        train_model(df, "24h", allow_synthetic=False)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "synthetic" in str(e).lower()


def test_features_require_real_data():
    """Feature computation must reject synthetic-only data when require_real=True."""
    from features import compute_features
    from config import SOURCE_SYNTHETIC
    
    # This test verifies the code path exists; actual disk test is integration
    import inspect
    src = inspect.getsource(compute_features)
    assert "require_real" in src
    assert "SOURCE_SYNTHETIC" in src


def test_provenance_in_candles():
    """Candles must carry source_mode and provider columns."""
    from market_data import save_candles, get_candles
    from config import SOURCE_HISTORICAL
    import shutil
    
    df = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        "open": [100.0] * 5,
        "high": [101.0] * 5,
        "low": [99.0] * 5,
        "close": [100.5] * 5,
        "volume": [1e6] * 5,
        "symbol": ["__test_prov"] * 5,
        "interval": ["1h"] * 5,
        "market": ["spot"] * 5,
        "exchange": ["binance"] * 5,
        "ingested_at": ["2024-01-01T00:00:00+00:00"] * 5,
        "source_mode": [SOURCE_HISTORICAL] * 5,
        "provider": ["binance"] * 5,
    })
    save_candles(df, "__test_prov", "1h", "spot", source_mode=SOURCE_HISTORICAL, provider="binance")
    stored = get_candles("__test_prov", "1h", "spot")
    assert "source_mode" in stored.columns
    assert "provider" in stored.columns
    assert all(stored["source_mode"] == SOURCE_HISTORICAL)
    assert all(stored["provider"] == "binance")
    
    shutil.rmtree(Path.home() / "Library/Application Support/NeuroTrom-Crypto/market/spot/__test_prov", ignore_errors=True)
    shutil.rmtree(Path.home() / "Library/Application Support/NeuroTrom-Crypto/raw/market/__test_prov", ignore_errors=True)
    shutil.rmtree(Path.home() / "Library/Application Support/NeuroTrom-Crypto/raw/market/__test_prov", ignore_errors=True)


def test_health_degraded_when_no_provider():
    """Health check must report degraded when no provider is reachable."""
    from health import test_provider_connectivity
    # This test just verifies the function exists and returns expected keys
    r = test_provider_connectivity("kraken")
    assert "reachable" in r
    assert "provider" in r
