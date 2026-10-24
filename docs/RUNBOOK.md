# Runbook — cloud-first

## Normal operation

GitHub Actions runs `.github/workflows/daily-research.yml` daily at 00:47 UTC (approximately 06:17 Asia/Kolkata), and also supports manual runs from the Actions tab. Each run fetches public market data, optionally inspects Kaggle history, retrains and evaluates the four research models, matures the prediction journal, and deploys the latest static JSON/CSV and dashboard to Pages.

- Dashboard: https://harshyadavv2456.github.io/NeuroTrom-Crypto/
- Latest JSON: https://harshyadavv2456.github.io/NeuroTrom-Crypto/data/latest.json
- Daily CSV: https://harshyadavv2456.github.io/NeuroTrom-Crypto/data/daily_update.csv
- Workflow: GitHub → Actions → Daily Crypto Refresh and GitHub Pages
- Manual refresh: choose the workflow and select **Run workflow**.

## One-time setup

1. Set the repository visibility to public after scanning tracked files and history for credentials and removing the historical FinVest parity document.
2. Under Settings → Pages, set the publishing source to GitHub Actions.
3. Enable Actions for the repository.
4. To use historical Kaggle input, set repository variable `KAGGLE_INPUT_DATASET` and configure secret `KAGGLE_API_TOKEN` or `KAGGLE_USERNAME` plus `KAGGLE_KEY`.
5. To version a separate Kaggle daily-snapshot dataset automatically, set `KAGGLE_OUTPUT_DATASET` to its exact owner/slug and `KAGGLE_OUTPUT_PUBLISH=true`. Never use the same dataset for input and output; the workflow blocks that configuration.
6. Review the input/output dataset licenses and actual candle timestamps, then manually run the daily workflow and inspect its full log.

## Kaggle freshness

The pipeline records both Kaggle metadata last-updated and latest candle timestamp. Data older than seven days is reported as stale. Stale hourly rows may only extend history before the API window; they are not substituted for current API prices.

The generated CSV is available as a static public file. If publishing it as a Kaggle dataset, review all applicable exchange and dataset licences first. Kaggle documents the ability to create datasets from public remote files and configure interval updates in the Kaggle Dataset Settings.

## Diagnosing a degraded run

Open the latest Actions run and inspect the actual exception, provider host tests, the Kaggle section and each model horizon. The dashboard intentionally remains published when it can generate a degraded status report. It must not claim that a blocked provider or stale candle data is current.

## Cost boundaries

The workflow uses standard GitHub-hosted Ubuntu runners and GitHub Pages (public repo on GitHub Free). It does not use a paid cloud server. Runtime per training run and GitHub account/monthly Actions limits still apply. A GitHub schedule is approximate, not a real-time service-level guarantee.
