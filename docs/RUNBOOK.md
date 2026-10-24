# Runbook

## Start API
```bash
cd ~/neurotrom-crypto
PYTHONPATH="" ~/neurotrom-crypto/.venv/bin/python -m uvicorn src.api:app --host 127.0.0.1 --port 8775
```

## Stop API
```bash
launchctl stop com.neurotrom.crypto.api
# or
pkill -f "uvicorn src.api:app"
```

## Backfill (synthetic fallback or real when reachable)
```bash
cd ~/neurotrom-crypto
PYTHONPATH="" ~/neurotrom-crypto/.venv/bin/python src/backfill.py
```

## Retrain (requires real data)
```bash
cd ~/neurotrom-crypto
PYTHONPATH="" ~/neurotrom-crypto/.venv/bin/python src/train.py
# or via API
curl -X POST http://127.0.0.1:8775/v1/crypto/train
```

## Health check (nonzero exit = degraded/blocked)
```bash
cd ~/neurotrom-crypto
PYTHONPATH="" ~/neurotrom-crypto/.venv/bin/python src/health.py
echo $?
```

## End-to-end verification
```bash
cd ~/neurotrom-crypto
PYTHONPATH="" ~/neurotrom-crypto/.venv/bin/python src/e2e.py
```

## Current state
- Dashboard: http://127.0.0.1:8775
- Overall status: BLOCKED — no real data provider reachable
- Data: 286,204 synthetic candles (SYNTHETIC_DEMO), 68 symbols
- Models: 4 LightGBM models trained on synthetic data (research-only)
- Predictions: 0 real (system correctly refuses to manufacture predictions)
- Providers: Binance unreachable; Kraken endpoint reachable via curl but Python requests gets ConnectionResetError

## Recovery
- Backfill is resumable: state is in `state/backfill_state.json`.
- If the API crashes, launchd restarts it.
- If the collector crashes, it reconnects with exponential backoff.
- When a provider becomes reachable from Python, run backfill to populate real data, then retrain.
