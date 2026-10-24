"""Data source registry, terms checker and provider adapters."""
from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

log = logging.getLogger(__name__)

DATA_SOURCES_PATH = Path(__file__).resolve().parent.parent.parent / "Library" / "Application Support" / "NeuroTrom-Crypto" / "state" / "sources.json"


@dataclass
class DataSource:
    id: str
    name: str
    provider: str
    data_type: str  # ohlcv, funding, open_interest, ticker, metadata
    endpoint_url: str
    docs_url: str
    auth_required: bool
    rate_limit: Optional[str] = None
    connection_limit: Optional[str] = None
    historical_coverage: str = ""
    update_frequency: str = ""
    quality_notes: str = ""
    usage_terms: str = ""
    commercial_status: str = "commercial_rights_unconfirmed"  # approved_for_personal_research, commercial_rights_unconfirmed, commercial_use_requires_license, approved_for_commercial_use
    conditions: str = ""
    terms_reviewed_date: str = ""
    approved_for_use: bool = False

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k != "approved_for_use"}

    @property
    def status(self) -> str:
        return self.commercial_status


# Approved sources for personal research and development
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
        connection_limit="No documented concurrent connection limit for REST",
        historical_coverage="Symbol-listing dependent; most pairs have several years of 1h/1d bars",
        update_frequency="Real-time with <1s delay",
        quality_notes="High quality, widely used. Some low-liquidity pairs have gaps.",
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
        connection_limit="Up to 1024 streams per connection, 5 connections per IP",
        historical_coverage="Same as REST",
        update_frequency="Real-time",
        quality_notes="Low latency. Must handle reconnects and resubscribe.",
        usage_terms="Same as Binance Spot REST API.",
        commercial_status="approved_for_personal_research",
        conditions="Personal/research use only.",
        terms_reviewed_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        approved_for_use=True,
    ),
    DataSource(
        id="binance_futures_rest",
        name="Binance USD-M Futures REST",
        provider="Binance",
        data_type="ohlcv",
        endpoint_url="https://fapi.binance.com/fapi/v1/klines",
        docs_url="https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md",
        auth_required=False,
        rate_limit="2400 request weight per minute per IP",
        connection_limit="No documented concurrent connection limit for REST",
        historical_coverage="Most perpetual pairs have 2+ years of history",
        update_frequency="Real-time",
        quality_notes="High quality. Perpetual contracts only; no inverse or coin-margined by default.",
        usage_terms="Same as Binance Spot REST API.",
        commercial_status="approved_for_personal_research",
        conditions="Personal/research use only.",
        terms_reviewed_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        approved_for_use=True,
    ),
    DataSource(
        id="binance_futures_ws",
        name="Binance USD-M Futures WebSocket",
        provider="Binance",
        data_type="ohlcv",
        endpoint_url="wss://fstream.binance.com/ws",
        docs_url="https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md",
        auth_required=False,
        rate_limit="Connection lifetime limits apply",
        connection_limit="Up to 1024 streams per connection, 5 connections per IP",
        historical_coverage="Same as REST",
        update_frequency="Real-time",
        quality_notes="Low latency. Funding rate streams available.",
        usage_terms="Same as Binance Spot REST API.",
        commercial_status="approved_for_personal_research",
        conditions="Personal/research use only.",
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
        rate_limit="~10-30 calls/min (free tier, varies)",
        connection_limit="Not documented",
        historical_coverage="Market cap, volume, price history for 10k+ coins",
        update_frequency="Minutes",
        quality_notes="Good for market cap and metadata. Rate limits are strict on free tier.",
        usage_terms="Free tier for personal/non-commercial use. Paid tier available for commercial use.",
        commercial_status="commercial_use_requires_license",
        conditions="Free tier: personal/research only. Commercial use requires CoinGecko paid plan.",
        terms_reviewed_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        approved_for_use=True,  # for personal research metadata only
    ),
    DataSource(
        id="ccxt",
        name="CCXT Unified Exchange API",
        provider="CCXT (MIT)",
        data_type="ohlcv",
        endpoint_url="https://github.com/ccxt/ccxt",
        docs_url="https://github.com/ccxt/ccxt/wiki/Manual",
        auth_required=False,
        rate_limit="Varies by exchange (each exchange has its own limits)",
        connection_limit="Varies by exchange",
        historical_coverage="Depends on exchange",
        update_frequency="Depends on exchange",
        quality_notes="Convenient unified interface. Still subject to each exchange's terms.",
        usage_terms="CCXT is MIT licensed, but underlying exchange data is subject to each exchange's terms of service.",
        commercial_status="commercial_rights_unconfirmed",
        conditions="Do not assume CCXT MIT license grants commercial data rights. Check each exchange's terms.",
        terms_reviewed_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        approved_for_use=True,  # as a convenience wrapper; rights depend on underlying exchange
    ),
]


def load_sources() -> list[DataSource]:
    if DATA_SOURCES_PATH.exists():
        try:
            data = json.loads(DATA_SOURCES_PATH.read_text())
            return [DataSource(**s) for s in data]
        except Exception:
            pass
    return list(APPROVED_SOURCES)


def save_sources(sources: list[DataSource]):
    DATA_SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_SOURCES_PATH.write_text(json.dumps([s.to_dict() for s in sources], indent=2))


def get_source(source_id: str) -> Optional[DataSource]:
    for s in load_sources():
        if s.id == source_id:
            return s
    return None


def approved_sources() -> list[DataSource]:
    return [s for s in load_sources() if s.approved_for_use]


def init():
    if not DATA_SOURCES_PATH.exists():
        save_sources(APPROVED_SOURCES)
        log.info("Initialized data sources registry with %d sources", len(APPROVED_SOURCES))


if __name__ == "__main__":
    init()
    for s in load_sources():
        print(f"{s.id}: {s.status} — {s.conditions[:80]}...")

