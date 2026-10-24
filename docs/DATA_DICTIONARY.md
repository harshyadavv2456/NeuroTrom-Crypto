# Data Dictionary

## market/{market}/{symbol}/{interval}.parquet
| Column | Type | Description |
|---|---|---|
| timestamp | datetime64[ns, UTC] | Candle open time |
| open | float64 | Opening price |
| high | float64 | High price |
| low | float64 | Low price |
| volume | float64 | Base asset volume |
| close | float64 | Closing price |
| symbol | str | Binance symbol (e.g., BTCUSDT) |
| interval | str | 1m, 1h, 1d |
| market | str | spot or futures |
| exchange | str | binance |
| ingested_at | str | ISO timestamp when written |

## journal/predictions.parquet
| Column | Type | Description |
|---|---|---|
| prediction_id | str | Unique ID |
| symbol | str | Asset |
| horizon | str | 1h, 4h, 24h, 7d |
| timestamp | str | Prediction creation time (UTC) |
| predicted_return | float64 | Model output |
| model_version | str | Model identifier |
| price_at_prediction | float64 | Price used |
| matured | bool | Outcome filled? |
| actual_return | float64 | Realised return |
| maturation_time | str | When outcome was recorded |
