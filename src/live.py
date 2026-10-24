"""Live market data collection via WebSocket + REST reconciliation."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import DATA, HISTORY_INTERVALS, MAX_LIVE_CONCURRENCY
from market_data import save_candles, get_candles, _ws_listen, rest_klines, fetch_ticker
from universe import universe_symbols

log = logging.getLogger(__name__)
STATE = DATA / "state"
LIVE_STATE = STATE / "live_state.json"
running = False
_collector_task: Optional[asyncio.Task] = None
_loop: Optional[asyncio.AbstractEventLoop] = None


def load_live_state() -> dict:
    if LIVE_STATE.exists():
        try:
            return json.loads(LIVE_STATE.read_text())
        except Exception:
            pass
    return {"running": False, "last_update": None, "streams": [], "errors": []}


def save_live_state(state: dict):
    LIVE_STATE.write_text(json.dumps(state, indent=2, default=str))


class LiveCollector:
    def __init__(self):
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._bars_1m: dict[str, list] = {}  # symbol -> list of 1m bar dicts
        self._last_flush: dict[str, datetime] = {}
    
    async def _on_kline(self, row: dict):
        """Callback for incoming kline data."""
        symbol = row["symbol"]
        interval = row["interval"]
        market = row.get("market", "spot")
        
        if interval == "1m":
            key = f"{symbol}_{market}"
            if key not in self._bars_1m:
                self._bars_1m[key] = []
            self._bars_1m[key].append(row)
            
            # Flush every 60 seconds or when bar closes
            now = datetime.now(timezone.utc)
            last = self._last_flush.get(key)
            if row.get("is_closed") or (last and (now - last).total_seconds() > 60):
                if self._bars_1m[key]:
                    df = pd.DataFrame(self._bars_1m[key])
                    df = df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time")
                    save_candles(df, symbol, "1m", market)
                    self._bars_1m[key] = []
                    self._last_flush[key] = now
    
    async def _run_spot_ws(self):
        symbols = universe_symbols()[:30]  # limit streams
        streams = [f"{s.lower()}@kline_1m" for s in symbols]
        await _ws_listen(streams, self._on_kline, market_type="spot")
    
    async def _run(self):
        self.running = True
        save_live_state({"running": True, "last_update": datetime.now(timezone.utc).isoformat(), "streams": [], "errors": []})
        
        tasks = [
            asyncio.create_task(self._run_spot_ws()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            log.error("Live collector error: %s", e)
        finally:
            self.running = False
            save_live_state({"running": False, "last_update": datetime.now(timezone.utc).isoformat(), "streams": [], "errors": []})
    
    def start(self):
        if self.running:
            log.warning("Live collector already running")
            return
        
        self.thread = threading.Thread(target=self._start_thread, daemon=True)
        self.thread.start()
    
    def _start_thread(self):
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._run())
        except Exception as e:
            log.error("Live collector thread error: %s", e)
    
    def stop(self):
        if self.loop and self.running:
            self.running = False
            for task in asyncio.all_tasks(self.loop):
                task.cancel()
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=10)


# Global collector instance
_collector = LiveCollector()


def start_live():
    _collector.start()
    log.info("Live collector started")


def stop_live():
    _collector.stop()
    log.info("Live collector stopped")


def get_live_status() -> dict:
    state = load_live_state()
    state["collector_running"] = _collector.running
    return state


def reconcile_gaps(max_gap_hours: int = 6):
    """Reconcile missing candles after downtime."""
    now = datetime.now(timezone.utc)
    symbols = universe_symbols()[:30]
    total = 0
    for sym in symbols:
        for interval in ["1h", "1d"]:
            last = get_last_timestamp(sym, interval, "spot")
            if last is None:
                continue
            age = (now - last).total_seconds() / 3600
            if age > max_gap_hours:
                try:
                    start = last - timedelta(hours=2)  # small overlap
                    rows = backfill_symbol(sym, interval, start, now, "spot")
                    total += rows
                except Exception as e:
                    log.error("Reconcile error %s %s: %s", sym, interval, e)
    return total


def get_last_timestamp(symbol: str, interval: str, market_type: str = "spot") -> Optional[datetime]:
    from market_data import get_last_timestamp as _glt
    return _glt(symbol, interval, market_type)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    print(get_live_status())

