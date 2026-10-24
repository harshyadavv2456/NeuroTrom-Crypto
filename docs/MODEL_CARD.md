# Model Card (auto-generated)

## LightGBM models
| Horizon | Trees | Leaves | Features | Status |
|---|---|---|---|---|
| 1h | 500 | ~15,500 | auto-detected | baseline |
| 4h | 500 | ~15,500 | auto-detected | baseline |
| 24h | 500 | ~15,500 | auto-detected | baseline |
| 7d | 500 | ~15,500 | auto-detected | baseline |

Settings: `learning_rate=0.025`, `num_leaves=31`, `min_data_in_leaf=300`, `feature_fraction=0.3`, `bagging_fraction=0.7`, `lambda_l2=100`, `extra_trees=True`. 500 rounds, 5 walk-forward folds, gap 5 bars.

## Candidate features (v1)
- Lagged log returns (1, 2, 3, 5, 10, 20, 60, 120, 240, 480 bars)
- Rolling volatility (10, 20, 60, 120 bars)
- RSI (5, 14)
- Moving-average distances (10, 20, 50, 100, 200)
- ATR (14)
- Volume ratios and changes
- Drawdown and range
- BTC 24h return and 60-bar beta (when BTC available)

## Known limits
- Personal research only; not validated for commercial use.
- 1h bars mean 24h horizon has ~20 independent observations per day per symbol.
- Fees assumed flat 10 bps; real costs vary by venue and size.
