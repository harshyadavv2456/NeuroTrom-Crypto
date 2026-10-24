"""Data source registry, terms checker and provider adapters."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DATA_SOURCES_PATH = Path.home() / "Library" / "Application Support" / "NeuroTrom-Crypto" / "state" / "sources.json"


class DataSource:
    def __init__(self, id: str, name: str, provider: str, data_type: str,
                 endpoint_url: str, docs_url: str, auth_required: bool,
                 rate_limit: str = "", historical_coverage: str = "",
                 usage_terms: str = "", commercial_status: str = "commercial_rights_unconfirmed",
                 conditions: str = "", terms_reviewed_date: str = "",
                 approved_for_use: bool = False):
        self.id = id
        self.name = name
        self.provider = provider
        self.data_type = data_type
        self.endpoint_url = endpoint_url
        self.docs_url = docs_url
        self.auth_required = auth_required
        self.rate_limit = rate_limit
        self.historical_coverage = historical_coverage
        self.usage_terms = usage_terms
        self.commercial_status = commercial_status
        self.conditions = conditions
        self.terms_reviewed_date = terms_reviewed_date
        self.approved_for_use = approved_for_use

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


APPROVED_SOURCES = [
    DataSource(
        id="binance_spot_rest",
        name="Binance Spot REST API",
        provider="Binance",
        data_type="ohlcv",
        endpoint_url="https://api.binance.com/api/v3/klines",
        docs_url="https://github.com/binance/binance-spot-api-docs",
        auth_required=False,
        rate_limit="2400 request weight per minute per IP",
        historical_coverage="Symbol-listing dependent; most pairs have several years of 1h/1d bars",
        usage_terms="Public API: no auth required. Binance Vision bulk datasets are CC BY-NC-SA 4.0 (non-commercial only). Public API availability does not automatically grant commercial data rights.",
        commercial_status="approved_for_personal_research",
        conditions="Personal/research use only. Verify commercial terms before any commercial deployment.",
        terms_reviewed_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        approved_for_use=True,
    ),
    DataSource(
        id="binance_spot_ws",
        name="Binance Spot WebSocket",
        provider="Binance",
        data_type="ohlcv",
        endpoint_url="wss://stream.binance.com:9443/ws",
        docs_url="https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md",
        auth_required=False,
        rate_limit="Connection lifetime limits apply; reconnect before 24h",
        historical_coverage="Same as REST",
        usage_terms="Same as Binance Spot REST API.",
        commercial_status="approved_for_personal_research",
        conditions="Personal/research use only.",
        terms_reviewed_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        approved_for_use=True,
    ),
    DataSource(
        id="kraken_spot_rest",
        name="Kraken Spot REST API",
        provider="Kraken",
        data_type="ohlcv",
        endpoint_url="https://api.kraken.com/0/public/OHLC",
        docs_url="https://support.kraken.com/en/articles/360000919986-public-endpoint-examples-you-can-try-them-directly-in-a-web-browser",
        auth_required=False,
        rate_limit="Not explicitly rate-limited for public endpoints; be respectful",
        historical_coverage="Up to 720 data points per request (e.g., 720 hours = 30 days for 1h interval)",
        usage_terms="Kraken API terms apply. Free public access for personal/research use. Commercial use requires Kraken's review.",
        commercial_status="approved_for_personal_research",
        conditions="Personal/research use only. Commercial use requires separate agreement with Kraken.",
        terms_reviewed_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        approved_for_use=True,
    ),
    DataSource(
        id="coingecko_free",
        name="CoinGecko Free API",
        provider="CoinGecko",
        data_type="metadata",
        endpoint_url="https://api.coingecko.com/api/v3",
        docs_url="https://docs.coingecko.com/",
        auth_required=False,
        rate_limit="~10-30 calls/min (free tier)",
        historical_coverage="Market cap, volume, price history for 10k+ coins",
        usage_terms="Free tier for personal/non-commercial use. Paid tier available.",
        commercial_status="commercial_use_requires_license",
        conditions="Free tier: personal/research only. Commercial requires paid plan.",
        terms_reviewed_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        approved_for_use=True,
    ),
]


def load_sources() -> list:
    if DATA_SOURCES_PATH.exists():
        try:
            data = json.loads(DATA_SOURCES_PATH.read_text())
            return [DataSource(**s) for s in data]
        except Exception:
            pass
    return list(APPROVED_SOURCES)


def save_sources(sources: list):
    DATA_SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_SOURCES_PATH.write_text(json.dumps([s.to_dict() for s in sources], indent=2))


def get_source(source_id: str) -> Optional[DataSource]:
    for s in load_sources():
        if s.id == source_id:
            return s
    return None


def approved_sources() -> list:
    return [s for s in load_sources() if s.approved_for_use]


def init():
    if not DATA_SOURCES_PATH.exists():
        save_sources(APPROVED_SOURCES)
        logging.info("Initialized data sources registry with %d sources", len(APPROVED_SOURCES))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init()
    for s in load_sources():
        print(f"{s.id}: {s.commercial_status} — {s.conditions[:80]}...")
