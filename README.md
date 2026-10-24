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
uv venv
uv pip install -e ".[dev]"
```

## Run

```bash
# Start API + dashboard
uv run uvicorn src.api:app --host 127.0.0.1 --port 8775

# Run health checks
uv run python src/health.py

# Backfill history
uv run python src/backfill.py

# Daily pipeline
uv run python src/daily.py

# End-to-end verification
uv run python src/e2e.py
```

## Tests

```bash
uv run pytest tests/ -v
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
