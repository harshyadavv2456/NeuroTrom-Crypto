# Model Card (generated from actual artifacts)

## LightGBM models
| Horizon | Trees | Leaves | Features | Provenance | Status |
|---|---|---|---|---|---|
| 1h | 500 | ~15,500 | auto-detected | Check provenance file | research-only until real data |
| 4h | 500 | ~15,500 | auto-detected | Check provenance file | research-only until real data |
| 24h | 500 | ~15,500 | auto-detected | Check provenance file | research-only until real data |
| 7d | 500 | ~15,500 | auto-detected | Check provenance file | research-only until real data |

Settings: `learning_rate=0.025`, `num_leaves=31`, `min_data_in_leaf=300`, `feature_fraction=0.3`, `bagging_fraction=0.7`, `lambda_l2=100`, `extra_trees=True`. 500 rounds, 5 walk-forward folds, gap 5 bars.

## Data provenance
All models record their training data provenance in `models/crypto_{horizon}_provenance.json`.
Values:
- `HISTORICAL_EXCHANGE` — trained on real historical market data
- `LIVE_EXCHANGE` — trained on live data (research-only)
- `SYNTHETIC_DEMO` — trained on synthetic data (not production-ready)

## Candidate features (v1)
- Lagged log returns (1, 2, 3, 5, 10, 20, 60, 120, 240, 480 bars)
- Rolling volatility (10, 20, 60, 120 bars)
- RSI (5, 14)
- Moving-average distances (10, 20, 50, 100, 200)
- ATR (14)
- Volume ratios and changes
- Range and drawdown
- BTC 24h return and 60-bar beta (when BTC available)

## Known limits
- Personal research only; not validated for commercial use.
- Models trained on synthetic data are labelled `SYNTHETIC_DEMO` and must not be used for real-market predictions.
- Real-data models require verified provider connectivity and historical ingestion.
