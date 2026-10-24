# Data Sources and Rights

## Data source policy

| Source | Role in cloud pipeline | Authentication | Public output policy |
|---|---|---|---|
| Binance Spot public REST | Hourly candles and 24-hour ticker for research models and forecasts | None | Non-commercial research only; do not assume commercial, redistribution, or signal-distribution rights |
| Kaggle dataset | Optional older hourly history for exact supported symbols | Kaggle token via GitHub Actions secret | Only explicit permissive licenses are accepted; source license/attribution must be checked |
| CoinDCX public REST | Not used by the public cloud pipeline | No credential required for public endpoints | Excluded because current CoinDCX API terms prohibit third-party display or redistribution of market data and derived research |

## Freshness rules

- Current ticker and hourly candle timestamps come from exchange responses, not the Kaggle dataset's last-updated date.
- Kaggle metadata freshness is reported independently. By default, datasets older than 7 days are marked stale, but compatible older history may only be used to extend dates before the 365-day API window.
- The pipeline accepts Kaggle input only if the declared license contains CC0, Public Domain, or CC BY 4.0 in a recognized form.
- Only exact symbol/quote matches with hourly OHLCV columns are accepted. Daily rows are not upsampled into fake hourly data. USD and USDT instruments are never silently relabelled.
- If live API access fails, the dashboard reports degraded status. Historical Kaggle data is not described as live data.

## Provider references

- Binance API endpoint documentation: https://github.com/binance/binance-spot-api-docs
- Binance public market-data-only endpoints: https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md
- Kaggle API documentation: https://www.kaggle.com/docs/api
- Kaggle dataset documentation: https://www.kaggle.com/docs/datasets
- CoinDCX API terms: https://docs.coindcx.com/ (sections 2.3(c) and 4)

## Important

This repository is for non-commercial research. A free public API endpoint does not automatically grant rights for paid products, trading signals, redistribution, or commercialization. The project does not execute trades, access balances, or store exchange account credentials.
