# Roadmap

## Phase 1 — Foundation (completed)
- [x] Project structure and isolation
- [x] Source registry with terms tracking
- [x] Provenance tracking (`SOURCE_LIVE`, `SOURCE_HISTORICAL`, `SOURCE_SYNTHETIC_DEMO`)
- [x] Resumable backfill with checkpoints
- [x] Universe discovery (68 liquid USDT pairs, synthetic fallback)
- [x] Feature engineering with `require_real` gate
- [x] LightGBM training (1h, 4h, 24h, 7d) — synthetic provenance only
- [x] Prediction journal with maturation
- [x] Local API + dashboard on `127.0.0.1:8775`
- [x] Health check with nonzero exit on failure
- [x] Daily pipeline (`catchup.py` + `daily.py`)
- [x] Launchd agents (api, collector, daily)
- [x] Tests (12/12 pass)
- [x] End-to-end verification (12/12 pass)

## Phase 2 — Real data ingestion (blocked)
- [ ] Verified reachable provider from Python runtime
- [ ] Historical backfill with real candles
- [ ] Live WebSocket collector with reconnects
- [ ] Funding rates, open interest, futures basis
- [ ] DuckDB analytics layer

## Phase 3 — Advanced models (not started)
- [ ] Neural network experiments (only if LightGBM is beaten on real data)
- [ ] Multi-exchange data (with rights verification)
- [ ] Ensemble with validated weights
- [ ] On-chain metrics (approved sources)

## Known blockers
- `api.binance.com` unreachable from this Python environment (connection timeout).
- `api.kraken.com` public endpoints respond to curl but Python `requests` receives `ConnectionResetError(54)` during TLS handshake. Network/firewall interception likely; not yet resolved.
- Binance Vision commercial rights unresolved; not used.
- System honestly reports `BLOCKED — no real data provider reachable` until a reliable Python-reachable source is available.
