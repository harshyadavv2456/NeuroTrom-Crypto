# Runbook

## Start
```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8775
```

## Stop
```bash
curl -X POST http://127.0.0.1:8775/v1/crypto/live/stop
# or
launchctl stop com.neurotrom.crypto.api
```

## Backfill
```bash
uv run python src/backfill.py
```

## Retrain
```bash
uv run python src/train.py
# or via API
curl -X POST http://127.0.0.1:8775/v1/crypto/train
```

## Health check
```bash
uv run python src/health.py
```

## Recovery
- Backfill is resumable: state is in `state/backfill_state.json`.
- If the API crashes, launchd restarts it.
- If the collector crashes, it reconnects with exponential backoff.
