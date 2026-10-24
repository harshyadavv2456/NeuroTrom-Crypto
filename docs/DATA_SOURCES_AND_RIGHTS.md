# Data Sources and Rights

## Approved sources

| Provider | Data Type | Auth | Rate Limit | Status | Conditions |
|---|---|---|---|---|---|
| Binance Spot REST | OHLCV | None | 2400 weight/min | `approved_for_personal_research` | Personal/research only; unreachable from current Python environment |
| Binance Spot WS | OHLCV live | None | 1024 streams/conn | `approved_for_personal_research` | Personal/research only; unreachable from current Python environment |
| Kraken Spot REST | OHLCV | None | Public endpoints | `approved_for_personal_research` | Personal/research only; endpoint reachable via curl but Python requests receives ConnectionResetError on this network |
| CCXT | OHLCV wrapper | Varies | Varies by exchange | `commercial_rights_unconfirmed` | Check each exchange's terms |
| CoinGecko Free | Metadata | None | ~10-30 req/min | `commercial_use_requires_license` | Paid plan needed for commercial use |

## Current network status (verified 2026-10-25)

- `api.binance.com`: unreachable from Python `requests` (connection timeout to port 443). Curl also times out.
- `api.kraken.com`: endpoint responds to curl (`/0/public/Time` and `/0/public/OHLC` return valid JSON). Python `requests` receives `ConnectionResetError(54)` on TLS handshake. Cause is network/firewall interception, not endpoint downtime.
- No commercial-use rights have been verified for any bulk historical dataset.

## Mandatory restrictions

- Binance Vision bulk datasets are CC BY-NC-SA 4.0 (non-commercial). We do NOT use Binance Vision.
- Public API availability does not automatically grant commercial data rights.
- No exchange provides unlimited, unrestricted API access.
- Rate limits are respected; no evasion or rotation.
- When a provider is unreachable, the system reports `BLOCKED — no real data provider reachable` and does not silently substitute another exchange's prices as if they were identical.

## Review date

All terms reviewed: 2026-10-25. Re-review before any commercial deployment.
