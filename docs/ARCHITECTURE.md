# Architecture

## Data flow
```
Binance REST/WS -> market_data.py -> raw/market/ + market/
               -> backfill.py (resumable, checkpointed)
               -> live.py (WebSocket, reconnect, dedup)
               -> features.py -> features/
               -> train.py -> models/
               -> journal.py -> journal/
               -> api.py -> :8775
               -> dashboard.html
```

## Storage layout
```
~/Library/Application Support/NeuroTrom-Crypto/
  raw/market/     — original provider responses
  market/         — normalized Parquet (timestamp, open, high, low, close, volume, symbol, interval, market, exchange, ingested_at)
  features/       — computed feature datasets
  models/         — LightGBM .txt models
  journal/        — predictions.parquet, outcomes.parquet, summary.json
  state/          — checkpoints, universe, backfill state, live state
  logs/           — application logs
  reports/        — training reports, e2e reports
```

## Key invariants
- All timestamps UTC.
- Completed bars only: no unfinished candle in features or labels.
- Walk-forward gap enforced between train/validation folds.
- A prediction is recorded before its outcome exists.
- API binds 127.0.0.1 only.
