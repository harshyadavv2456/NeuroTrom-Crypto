# Roadmap

## Phase 1 — Foundation (current)
- [x] Project structure and isolation
- [x] Binance REST + WebSocket adapters
- [x] Resumable backfill with checkpoints
- [x] Universe discovery (60 liquid USDT pairs)
- [x] Feature engineering (price, volume, BTC regime)
- [x] LightGBM training (1h, 4h, 24h, 7d)
- [x] Prediction journal with maturation
- [x] Local API + dashboard
- [x] Health checks
- [x] Daily pipeline
- [x] Launchd agents
- [x] Tests

## Phase 2 — Reliability
- [ ] Funding rates collector
- [ ] Open interest collector
- [ ] Futures basis features
- [ ] Order-book imbalance (limited set)
- [ ] DuckDB analytics layer
- [ ] Automated gap detection and backfill after sleep

## Phase 3 — Advanced models
- [ ] Neural network experiments (only if LightGBM is beaten)
- [ ] Multi-exchange data (with rights verification)
- [ ] Ensemble with validated weights
- [ ] On-chain metrics (approved sources)

## Known blockers
- Binance Vision commercial rights unresolved; not used.
- Fine-grain (1m) data for 15+ symbols is large; bounded by config.
- Funding rates require separate endpoint; implemented in Phase 2.
