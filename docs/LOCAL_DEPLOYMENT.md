# Cloud Deployment

NeuroTrom Crypto now uses GitHub Actions and a static GitHub Pages frontend. The former macOS launchd/API deployment is not required for the public dashboard.

## Required one-time settings

1. Repository visibility: public (after checking and cleaning Git history).
2. Pages: Settings → Pages → Build and deployment → Source: GitHub Actions.
3. Actions: enabled for the repository.

## Workflow

- Workflow: `.github/workflows/daily-research.yml`
- Schedule: daily at 00:47 UTC, approximately 06:17 Asia/Kolkata.
- Manual run: GitHub → Actions → Daily Crypto Refresh and GitHub Pages → Run workflow.
- Runner: GitHub-hosted Ubuntu.
- Static output directory: `site/`
- Dashboard output: `site/data/latest.json`
- Price-chart output: `site/data/history.json`
- Daily CSV output: `site/data/daily_update.csv`

## Optional Kaggle input

Set Actions variable `KAGGLE_INPUT_DATASET` to an exact dataset slug, such as `owner/dataset-slug`. Add either secret `KAGGLE_API_TOKEN` or both secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Only licenses in the pipeline's explicit permissive allowlist and exact hourly USDT candles are eligible. Dataset metadata freshness is reported separately. A dataset that has not been updated recently must not be mistaken for current market data.

## Diagnostics

If a scheduled run fails, open its job log and inspect the on-page `Data & automation` section. The workflow is intentionally allowed to publish a degraded report when a provider or a model horizon is unavailable. It must not claim that historical-only data is fresh live data.

## Kaggle remote snapshot

Once Pages works, the generated CSV is available at:

`https://harshyadavv2456.github.io/NeuroTrom-Crypto/data/daily_update.csv`

A Kaggle dataset can be created from this remote file and configured for daily refresh. This is an output dataset and is independent from optional Kaggle historical input.
