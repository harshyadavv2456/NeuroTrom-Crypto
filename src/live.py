"""Persistent live market collector with REST polling fallback.

The collector is a standalone launchd process and can also be started/stopped by
the local API. It records only observed exchange responses; it never fabricates
candles or calls synthetic fixture data "live".
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DATA, SOURCE_LIVE, SOURCE_HISTORICAL, SOURCE_SYNTHETIC, UNIVERSE_SIZE
from market_data import (
    fetch_ticker,
    test_provider_connectivity,
    http_get,
    save_candles,
    kraken_ohlc,
    rest_klines,
)
from universe import load_universe, build_universe

log = logging.getLogger(__name__)
STATE = DATA / "state"
LIVE_STATE = STATE / "live_state.json"
TICKER_ROOT = DATA / "market_tickers"
POLL_INTERVAL_SECONDS = 60
MIN_REQUEST_INTERVAL_SECONDS = 0.75
STATUS_STALE_SECONDS = 180
CANDLE_REFRESH_SECONDS = 900


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def load_live_state() -> dict:
    try:
        return json.loads(LIVE_STATE.read_text())
    except (OSError, ValueError):
        return {
            "running": False,
            "last_update": None,
            "streams": [],
            "errors": [],
            "provider_status": {},
        }


def save_live_state(state: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = LIVE_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, LIVE_STATE)


def _save_ticker_snapshot(ticker: dict, market_meta: dict) -> None:
    """Append a real ticker response to a per-provider Parquet log."""
    provider = str(ticker["provider"]).lower()
    symbol = str(ticker["symbol"]).upper()
    folder = TICKER_ROOT / provider
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{symbol}.parquet"

    received_at = _now()
    row = {
        "symbol": symbol,
        "provider": provider,
        "exchange": provider,
        "market": str(market_meta.get("market", "spot")),
        "price": float(ticker["price"]),
        "change_pct": ticker.get("change_pct"),
        "volume_24h": float(ticker["volume"]) if ticker.get("volume") is not None else None,
        "high_24h": float(ticker["high"]) if ticker.get("high") is not None else None,
        "low_24h": float(ticker["low"]) if ticker.get("low") is not None else None,
        "provider_timestamp": ticker.get("timestamp"),
        "received_at": received_at.isoformat(),
        "source_mode": SOURCE_LIVE,
        "transport": ticker.get("transport", "requests"),
    }
    fresh = pd.DataFrame([row])
    if path.exists():
        try:
            old = pd.read_parquet(path)
            fresh = pd.concat([old, fresh], ignore_index=True)
            if "received_at" in fresh.columns:
                fresh = fresh.drop_duplicates(subset=["received_at"], keep="last")
                fresh["received_at"] = pd.to_datetime(fresh["received_at"], utc=True, errors="coerce")
                fresh = fresh.sort_values("received_at").tail(10000)
                fresh["received_at"] = fresh["received_at"].astype(str)
        except Exception as exc:
            log.warning("Could not read ticker history %s: %s", path, exc)
    tmp = path.with_suffix(".parquet.tmp")
    fresh.to_parquet(tmp, index=False, engine="pyarrow")
    os.replace(tmp, path)


def _real_universe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "symbol" not in df.columns or "exchange" not in df.columns:
        return pd.DataFrame()
    result = df[df["exchange"].astype(str).str.lower().isin(["binance", "kraken"])].copy()
    if "universe_provenance" in result.columns:
        result = result[result["universe_provenance"].astype(str).isin([SOURCE_LIVE, SOURCE_HISTORICAL])]
    if "status" in result.columns:
        result = result[~result["status"].astype(str).str.upper().isin(["DEMO_ONLY", "SYNTHETIC"])]
    return result.reset_index(drop=True)


def _probe_providers() -> dict:
    return {
        "binance": test_provider_connectivity("binance"),
        "kraken": test_provider_connectivity("kraken"),
    }


def _refresh_universe_if_needed(provider_status: dict) -> pd.DataFrame:
    current = load_universe()
    current_real = _real_universe(current)
    reachable = {
        name for name, result in provider_status.items()
        if result.get("reachable", False)
    }
    current_exchanges = set(current_real["exchange"].astype(str).str.lower()) if not current_real.empty else set()

    # Do not overwrite a known-good real universe while every provider is down.
    must_refresh = current_real.empty or not current_exchanges.intersection(reachable)
    if reachable and must_refresh:
        try:
            refreshed = build_universe(UNIVERSE_SIZE)
            refreshed_real = _real_universe(refreshed)
            if not refreshed_real.empty:
                log.info("Refreshed real universe: %d symbols from %s",
                         len(refreshed_real), sorted(set(refreshed_real["exchange"].astype(str).str.lower())))
                return refreshed_real
        except Exception:
            log.exception("Could not refresh the real exchange universe")

    if not current_real.empty:
        # Keep only symbols whose assigned provider is currently reachable.
        return current_real[current_real["exchange"].astype(str).str.lower().isin(reachable)].reset_index(drop=True)
    return pd.DataFrame()


class LiveCollector:
    def __init__(self) -> None:
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_candle_refresh = 0.0

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive() and not self.stop_event.is_set())

    def start(self) -> None:
        state = load_live_state()
        state["desired_state"] = "running"
        save_live_state(state)
        with self._lock:
            if self.thread and self.thread.is_alive():
                self.stop_event.clear()
                log.info("Collector thread is already running")
                return
            # If launchd owns the process, update desired_state only; the process
            # already holding the cross-process lock will resume on its next cycle.
            owner_pid = state.get("pid")
            if owner_pid and _pid_alive(owner_pid):
                log.info("Collector process %s already owns the runtime; requested start", owner_pid)
                return
            self.stop_event.clear()
            self.thread = threading.Thread(
                target=self.run_forever,
                name="neurotrom-crypto-live-collector",
                daemon=True,
            )
            self.thread.start()

    def stop(self) -> None:
        # A persisted control state works for either the launchd process or the
        # API-managed thread. KeepAlive can restart a process, but it will remain
        # paused until the user explicitly starts it again.
        state = load_live_state()
        state["desired_state"] = "paused"
        state["status"] = "paused"
        state["streams"] = []
        save_live_state(state)

    def _update_status(self, **updates) -> None:
        state = load_live_state()
        state.update(updates)
        state["collector_running"] = self.running
        state["updated_at"] = _now_iso()
        save_live_state(state)

    def _refresh_recent_candles(self, universe: pd.DataFrame) -> None:
        """Occasionally reconcile the latest closed 1h bar from its actual exchange."""
        now_mono = time.monotonic()
        if now_mono - self._last_candle_refresh < CANDLE_REFRESH_SECONDS:
            return
        self._last_candle_refresh = now_mono
        for _, meta in universe.head(20).iterrows():
            if self.stop_event.is_set():
                return
            symbol = str(meta["symbol"])
            provider = str(meta["exchange"]).lower()
            try:
                if provider == "binance":
                    candles = rest_klines(symbol, "1h", limit=5)
                    if not candles.empty:
                        save_candles(candles, symbol, "1h", "spot",
                                     source_mode=SOURCE_HISTORICAL, provider=provider)
                elif provider == "kraken":
                    candles = kraken_ohlc(symbol, interval=60, symbol=symbol, interval_label="1h")
                    if not candles.empty:
                        save_candles(candles, symbol, "1h", "spot",
                                     source_mode=SOURCE_HISTORICAL, provider=provider)
            except Exception as exc:
                log.warning("Recent candle refresh failed for %s via %s: %s", symbol, provider, exc)
            self.stop_event.wait(MIN_REQUEST_INTERVAL_SECONDS)

    def _cycle(self) -> tuple[int, list[str], dict]:
        providers = _probe_providers()
        universe = _refresh_universe_if_needed(providers)
        if universe.empty:
            errors = [
                f"{name}: {value.get('error', 'unreachable')}"
                for name, value in providers.items()
                if not value.get("reachable", False)
            ]
            state = load_live_state()
            self._update_status(
                running=True,
                collector_running=True,
                provider_status=providers,
                streams=[],
                last_update=state.get("last_update"),
                errors=errors or ["No real exchange symbols available"],
                status="blocked",
                source_mode=None,
            )
            return 0, errors, providers

        ok_symbols: list[str] = []
        errors: list[str] = []
        latest = None
        failures_in_a_row = 0

        for _, meta in universe.iterrows():
            if self.stop_event.is_set():
                break
            symbol = str(meta["symbol"])
            provider = str(meta["exchange"]).lower()
            if not providers.get(provider, {}).get("reachable", False):
                continue
            try:
                ticker = fetch_ticker(symbol, "spot", provider)
                _save_ticker_snapshot(ticker, meta.to_dict())
                received = _now_iso()
                ok_symbols.append(symbol)
                latest = received
                failures_in_a_row = 0
                self._update_status(
                    running=True,
                    collector_running=True,
                    status="live" if not errors else "degraded",
                    last_update=received,
                    streams=ok_symbols[-100:],
                    provider_status=providers,
                    errors=errors[-20:],
                    last_symbol=symbol,
                    last_provider=provider,
                    source_mode=SOURCE_LIVE,
                )
            except Exception as exc:
                message = f"{provider}/{symbol}: {type(exc).__name__}: {str(exc)[:220]}"
                errors.append(message)
                failures_in_a_row += 1
                log.warning("Ticker failed: %s", message)
                if failures_in_a_row >= 5:
                    log.error("Five consecutive ticker failures; ending this cycle to respect provider limits")
                    break
            self.stop_event.wait(MIN_REQUEST_INTERVAL_SECONDS)

        # Ticker snapshots are real, but are not candles. Refresh completed exchange
        # candles separately and keep their HISTORICAL_EXCHANGE provenance.
        if ok_symbols and not self.stop_event.is_set():
            self._refresh_recent_candles(universe)

        previous = load_live_state().get("last_update")
        self._update_status(
            running=True,
            collector_running=True,
            status="live" if ok_symbols and not errors else ("degraded" if ok_symbols else "blocked"),
            last_update=latest or previous,
            streams=ok_symbols[-100:],
            provider_status=providers,
            errors=errors[-50:],
            universe_count=len(universe),
            source_mode=SOURCE_LIVE if ok_symbols else None,
        )
        return len(ok_symbols), errors, providers

    def run_forever(self) -> None:
        """Blocking single-instance worker used by launchd and the API."""
        STATE.mkdir(parents=True, exist_ok=True)
        lock_handle = (STATE / "live_collector.lock").open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("Another NeuroTrom Crypto collector already owns the lock")
            lock_handle.close()
            return

        prior = load_live_state()
        desired = prior.get("desired_state", "running")
        self.stop_event.clear()
        log.info("NeuroTrom Crypto live collector starting (pid=%s)", os.getpid())
        save_live_state({
            **prior,
            "pid": os.getpid(),
            "running": desired != "paused",
            "collector_running": True,
            "status": "paused" if desired == "paused" else "starting",
            "started_at": _now_iso(),
            "last_update": prior.get("last_update"),
            "streams": [] if desired == "paused" else prior.get("streams", []),
            "errors": prior.get("errors", []),
            "provider_status": prior.get("provider_status", {}),
            "desired_state": desired,
        })
        try:
            while not self.stop_event.is_set():
                state = load_live_state()
                if state.get("desired_state", "running") == "paused":
                    self._update_status(running=False, collector_running=True, status="paused", streams=[])
                    self.stop_event.wait(2.0)
                    continue

                started = time.monotonic()
                try:
                    count, errors, providers = self._cycle()
                    log.info("Collector cycle: %d real ticker snapshots; %d errors; providers=%s",
                             count, len(errors), {k: v.get("reachable") for k, v in providers.items()})
                except Exception as exc:
                    log.exception("Collector cycle failed: %s", exc)
                    self._update_status(status="degraded", errors=[f"{type(exc).__name__}: {str(exc)[:300]}"])
                elapsed = time.monotonic() - started
                self.stop_event.wait(max(2.0, POLL_INTERVAL_SECONDS - elapsed))
        finally:
            state = load_live_state()
            state.update({
                "running": False,
                "collector_running": False,
                "status": "stopped",
                "stopped_at": _now_iso(),
                "pid": None,
            })
            save_live_state(state)
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()
            log.info("NeuroTrom Crypto live collector stopped")


_collector = LiveCollector()


def _pid_alive(value) -> bool:
    try:
        pid = int(value)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError, OSError):
        return False


def start_live() -> None:
    _collector.start()


def stop_live() -> None:
    _collector.stop()


def get_live_status() -> dict:
    state = load_live_state()
    owner_alive = _pid_alive(state.get("pid"))
    state["collector_running"] = _collector.running or owner_alive
    state["running"] = state["collector_running"] and state.get("desired_state", "running") != "paused"
    if state.get("last_update"):
        try:
            stamp = pd.Timestamp(state["last_update"])
            stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
            age = (pd.Timestamp.now(tz="UTC") - stamp).total_seconds()
            state["last_update_age_seconds"] = round(max(age, 0), 1)
            state["fresh"] = age <= STATUS_STALE_SECONDS and bool(state.get("streams"))
        except Exception:
            state["fresh"] = False
    else:
        state["last_update_age_seconds"] = None
        state["fresh"] = False
    return state


def reconcile_gaps(max_gap_hours: int = 6):
    """Reconcile only eligible real exchange candles; never treat demo bars as coverage."""
    from market_data import get_candles, backfill_symbol
    from universe import load_universe

    now = _now()
    universe = _real_universe(load_universe())
    if universe.empty:
        return {"reconciled": 0, "errors": ["no real exchange universe"]}

    reconciled = 0
    errors = []
    for _, meta in universe.head(20).iterrows():
        symbol = str(meta["symbol"])
        provider = str(meta["exchange"]).lower()
        for interval in ("1h", "1d"):
            try:
                history = get_candles(symbol, interval, "spot", source_mode=SOURCE_HISTORICAL)
                if history.empty:
                    start = now - pd.Timedelta(days=30 if interval == "1h" else 365)
                else:
                    last = pd.to_datetime(history["timestamp"], utc=True).max()
                    if (now - last).total_seconds() <= max_gap_hours * 3600:
                        continue
                    start = last.to_pydatetime() - pd.Timedelta(hours=2)
                rows = backfill_symbol(symbol, interval, start, now, "spot", provider=provider)
                reconciled += int(rows or 0)
            except Exception as exc:
                errors.append(f"{symbol} {interval} via {provider}: {exc}")
    return {"reconciled": reconciled, "errors": errors, "status": "complete" if not errors else "degraded"}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _collector.run_forever()
