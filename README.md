# NeuroTrom Crypto

Independent local crypto-market intelligence engine. Learns from NeuroTrom FinVest patterns but operates on its own data, models, and services.

**Data:** Binance public APIs (personal/research use only).  
**Models:** LightGBM for 1h/4h/24h/7d horizons.  
**Storage:** `~/Library/Application Support/NeuroTrom-Crypto/`  
**API:** `http://127.0.0.1:8775`  
**Dashboard:** `http://127.0.0.1:8775/`

## Setup

```bash
cd ~/neurotrom-crypto
uv sync --no-install-project --group dev
```

## Run

```bash
# Start API + dashboard
uv run --no-sync uvicorn --app-dir src api:app --host 127.0.0.1 --port 8775

# Verify source connectivity and current real/demo provenance
uv run --no-sync python src/health.py

# First real-data bootstrap: reachable exchange -> historical candles -> train four real models -> seed forward journal
# This never falls back to synthetic data; it exits nonzero if the provider or any required stage is unavailable.
uv run --no-sync python src/bootstrap_real.py

# Resume/backfill the available real exchange history
uv run --no-sync python src/backfill.py

# Daily orchestrated pipeline
uv run --no-sync python src/daily.py

# End-to-end verification
uv run --no-sync python src/e2e.py
```

## Tests

```bash
uv run --no-sync pytest tests/ -v
```

## Auto-start (launchd)

```bash
cp ~/neurotrom-crypto/launchd/com.neurotrom.crypto.*.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.neurotrom.crypto.api.plist
launchctl load ~/Library/LaunchAgents/com.neurotrom.crypto.collector.plist
launchctl load ~/Library/LaunchAgents/com.neurotrom.crypto.daily.plist
```

## License

MIT (source code). Data provider terms are separate; see `docs/DATA_SOURCES_AND_RIGHTS.md`.
