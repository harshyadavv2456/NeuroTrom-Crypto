# Data Sources and Rights

## Approved sources

| Provider | Data Type | Auth | Rate Limit | Status | Conditions |
|---|---|---|---|---|---|
| Binance Spot REST | OHLCV | None | 2400 weight/min | `approved_for_personal_research` | Personal/research only |
| Binance Spot WS | OHLCV live | None | 1024 streams/conn | `approved_for_personal_research` | Personal/research only |
| Binance Futures REST | OHLCV | None | 2400 weight/min | `approved_for_personal_research` | Personal/research only |
| Binance Futures WS | OHLCV live | None | 1024 streams/conn | `approved_for_personal_research` | Personal/research only |
| CoinGecko Free | Metadata | None | ~10-30 req/min | `commercial_use_requires_license` | Paid plan needed for commercial use |
| CCXT | OHLCV wrapper | Varies | Varies by exchange | `commercial_rights_unconfirmed` | Check each exchange's terms |

## Mandatory restrictions

- Binance Vision bulk datasets are CC BY-NC-SA 4.0 (non-commercial). We do NOT use Binance Vision.
- Public API availability does not automatically grant commercial data rights.
- No exchange provides unlimited, unrestricted API access.
- Rate limits are respected; no evasion or rotation.

## Review date

All terms reviewed: 2026-10-25. Re-review before any commercial deployment.
