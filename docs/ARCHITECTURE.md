# Architecture — cloud-first

## Deployment

The primary product is a static site deployed to GitHub Pages. No Mac, launchd agent, local API server, persistent server process or paid cloud host is required.

```
GitHub Actions (daily 00:47 UTC)
        |
        v
Public Binance Spot REST API -----> 24h tickers
        |                            completed 1h candles
        v
Optional Kaggle historical data
  (license + freshness + schema checks)
        |
        v
Normalize exact-symbol hourly OHLCV
        |
        v
Lagged return/volatility/RSI features
        |
        v
LightGBM models: 1h / 4h / 24h / 7d
        |
        v
Time-ordered holdout evaluation
        |
        v
Forecast journal + realized-outcome maturation
        |
        v
site/data/latest.json + history.json + journal.json + daily_update.csv
        |
        v
Static site artifact -> GitHub Pages
```

## Data safety invariants

- Binance Spot endpoints are public and unauthenticated; this workflow never uses private account or trading endpoints.
- Timestamps are UTC and current prices come from the current API call, not from Kaggle metadata or old Kaggle candles.
- Kaggle input must pass source licence allowlisting, exact-symbol and exact USDT quote checks, numeric OHLCV parsing, and hourly-cadence validation.
- Compatible Kaggle history can extend dates older than the API window; it is never labelled as current live data.
- The actual latest timestamp inside the dataset is checked independently from Kaggle's metadata last-updated timestamp, so edited dataset metadata cannot disguise stale candles.
- Completed bars only; unfinished hourly candles are removed before feature engineering and training.
- Four separate LightGBM models use time-ordered train/holdout splits. Holdout metrics are research diagnostics, not claims of profitability.
- The forecast journal is retained in the generated Pages JSON and matured only when a later bar reaches the forecast horizon.
- If provider access, sufficient data or model training fails, the dashboard publishes DEGRADED status with warnings rather than pretending the pipeline is healthy.
- No live orders, wallets or exchange credentials are used.
- CoinDCX public data is not part of the public site because its API terms restrict third-party display/redistribution of market data and derived research.
