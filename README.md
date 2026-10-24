# NeuroTrom Crypto

Automated, non-commercial crypto quantitative research. The cloud workflow fetches public market data, checks optional Kaggle history, retrains LightGBM models, evaluates them on a time-ordered holdout, generates forecast snapshots, and deploys a static dashboard to GitHub Pages.

**No Mac service, API server, wallet, or trading credential is required.** The scheduled workflow runs on GitHub-hosted Ubuntu runners.

## Outputs

- Dashboard: `https://harshyadavv2456.github.io/NeuroTrom-Crypto/` (after Pages is enabled)
- Latest report: `site/data/latest.json`
- Last 7 days of hourly close series: `site/data/history.json`
- Kaggle-compatible daily snapshot: `site/data/daily_update.csv`
- Run summary: `site/data/run_history.json`

These are compact research outputs, not a raw historical exchange-data dump. The model estimates are experimental, not trading signals or evidence of profitability.

## Daily automation

`.github/workflows/daily-research.yml` schedules the data/model pipeline for approximately **06:17 Asia/Kolkata every day** (00:47 UTC) and supports manual runs from the Actions tab. It:

1. Checks Binance's documented public market-data REST hosts, with failover.
2. Fetches up to 365 days of completed hourly candles for BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT and BNBUSDT, plus current 24-hour tickers.
3. Optionally checks and downloads a configured Kaggle dataset. The pipeline inspects its update timestamp and license, and only accepts exact-symbol, hourly USDT OHLCV rows from an explicit permissive-license allowlist. Compatible Kaggle rows may extend history **before** the fetched API window; stale, daily-only, quote-mismatched or unknown-license data is reported and not silently treated as current.
4. Retrains 1h, 4h, 24h and 7d LightGBM models from the verified hourly series, checks time-ordered holdout metrics, and creates forecasts only when usable data and models exist.
5. Publishes JSON/CSV results to GitHub Pages even when a run is degraded, so the dashboard reports the failure instead of showing old data as current.

CoinDCX API data is deliberately excluded from public-page output: its currently published API terms prohibit third-party display or redistribution of market data and derived research. No exchange account API keys or trading endpoints are used by this cloud pipeline.

## One-time GitHub setup

1. Make the repository public and enable **Settings → Pages → Build and deployment → Source: GitHub Actions**. GitHub Pages is available for public repositories on GitHub Free.
2. Ensure Actions are enabled. Run **Actions → Daily Crypto Refresh and GitHub Pages → Run workflow** once to verify the first deployment.
3. The schedule runs from the default branch; a workflow may be delayed by GitHub load.

### Optional Kaggle historical input

If you have a particular dataset whose license permits this use, set the repository variable:

- `KAGGLE_INPUT_DATASET` = `owner/dataset-slug`

Add either a Kaggle API token as the Actions secret `KAGGLE_API_TOKEN`, or the legacy `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets. Never commit a token or `kaggle.json`. The job skips Kaggle ingestion when authentication is absent, the license is not in the allowlist, or files are incompatible.

A dataset's Kaggle "last updated" time is reported separately from the latest market-candle timestamp. An old dataset may still be historical archive data; it is never used to claim that live prices are current.

### Optional auto-updated Kaggle snapshot

The workflow generates `site/data/daily_update.csv` every run. After GitHub Pages is deployed, you can create a Kaggle dataset using the remote-file/GitHub dataset connector and point it at:

`https://harshyadavv2456.github.io/NeuroTrom-Crypto/data/daily_update.csv`

Set the Kaggle dataset's interval refresh to daily. Kaggle documents interval updates for datasets created from GitHub or remote files. This is separate from the optional historical input above.

## Run and test locally

Python 3.12+ is required. This cloud-only pipeline uses environment-local output paths and never starts the legacy macOS launchd agents.

```bash
uv sync --no-install-project --group dev
uv run --no-sync python scripts/cloud_pipeline.py
PYTHONPATH=src uv run --no-sync pytest -q tests
```

Kaggle is optional; without a configured source the pipeline uses the Binance public market-data API.

## Data and licensing

- Binance Spot public REST is used for non-commercial research only. Review the provider's applicable terms before public commercialization, paid signals, redistribution, or production trading.
- Kaggle data is only ingested when its dataset license is explicitly accepted by this repository's allowlist. The dataset owner is responsible for ensuring the dataset's provenance and license are correct.
- This project is not investment advice. It does not place orders or connect to user balances/wallets.
- Public market feeds can fail, be rate-limited, or change. All output includes run and freshness information; degraded output should not be interpreted as a current forecast.

See [Data Sources and Rights](docs/DATA_SOURCES_AND_RIGHTS.md) and [Cloud Deployment](docs/LOCAL_DEPLOYMENT.md).
