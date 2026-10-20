"""
Financial Contagion Monitor v2 — src/data/cache.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

LiveCache — SQLite-backed cache for yfinance live market data.
All downstream modules read from cache, never directly from yfinance.
Polling interval: 5 minutes. Stale threshold: 24 hours.

If yfinance is temporarily unavailable, the dashboard continues showing
the last known data with a "DATA STALE" badge on the affected node.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

POLL_INTERVAL_SECONDS  = int(os.getenv("YFINANCE_POLL_INTERVAL", "300"))   # 5 min
STALE_THRESHOLD_SECONDS = int(os.getenv("YFINANCE_STALE_THRESHOLD", "86400"))  # 24 hr

CREATE_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS live_cache (
    ticker          TEXT PRIMARY KEY,
    bank_name       TEXT NOT NULL,
    close_price     REAL,
    daily_return    REAL,
    volume          REAL,
    volatility_10d  REAL,
    mapped_features TEXT,
    last_updated    TEXT,
    is_stale        INTEGER DEFAULT 0
);
"""


class LiveCache:
    """
    Read/write cache for yfinance ticker data.

    Usage:
        cache = LiveCache("data/cache.db")
        cache.initialize()
        cache.set("SBIN.NS", "State Bank of India (SBI)", data_dict)
        result = cache.get("SBIN.NS")
        if result: print(result["close_price"])
    """

    def __init__(self, db_path: str = "data/cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    def initialize(self) -> None:
        """Creates cache table if not exists. Sets WAL mode."""
        conn = self._connect()
        conn.executescript(CREATE_CACHE_TABLE)
        conn.commit()
        conn.close()

    def get(self, ticker: str) -> Optional[dict]:
        """
        Returns cached data for ticker if fresher than POLL_INTERVAL_SECONDS.
        Returns None if no data or data is too old (triggers re-fetch).
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM live_cache WHERE ticker = ?", (ticker,)
        ).fetchone()
        conn.close()

        if row is None:
            return None

        data = dict(zip([
            "ticker", "bank_name", "close_price", "daily_return",
            "volume", "volatility_10d", "mapped_features",
            "last_updated", "is_stale"
        ], row))

        # Check freshness
        if data.get("last_updated"):
            try:
                last_updated = datetime.fromisoformat(data["last_updated"])
                age_seconds = (datetime.now() - last_updated).total_seconds()
                if age_seconds > POLL_INTERVAL_SECONDS:
                    return None  # too old — caller should re-fetch
            except ValueError:
                return None

        # Deserialize mapped_features JSON blob
        if data.get("mapped_features"):
            try:
                data["mapped_features"] = json.loads(data["mapped_features"])
            except (json.JSONDecodeError, TypeError):
                data["mapped_features"] = {}

        return data

    def get_stale(self, ticker: str) -> Optional[dict]:
        """
        Returns cached data regardless of age.
        Used as fallback when yfinance is unavailable.
        Caller should mark the result with a DATA STALE badge.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM live_cache WHERE ticker = ?", (ticker,)
        ).fetchone()
        conn.close()

        if row is None:
            return None

        data = dict(zip([
            "ticker", "bank_name", "close_price", "daily_return",
            "volume", "volatility_10d", "mapped_features",
            "last_updated", "is_stale"
        ], row))

        if data.get("mapped_features"):
            try:
                data["mapped_features"] = json.loads(data["mapped_features"])
            except (json.JSONDecodeError, TypeError):
                data["mapped_features"] = {}

        return data

    def set(self, ticker: str, bank_name: str, data: dict) -> None:
        """
        Stores ticker data with current timestamp.

        Args:
            ticker:    NSE ticker e.g. "SBIN.NS"
            bank_name: Full institution_id e.g. "State Bank of India (SBI)"
            data:      Dict with keys: close_price, daily_return, volume,
                       volatility_10d, mapped_features (dict)
        """
        mapped_json = json.dumps(data.get("mapped_features", {}))
        now = datetime.now().isoformat()

        conn = self._connect()
        conn.execute("""
            INSERT INTO live_cache
                (ticker, bank_name, close_price, daily_return, volume,
                 volatility_10d, mapped_features, last_updated, is_stale)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(ticker) DO UPDATE SET
                close_price     = excluded.close_price,
                daily_return    = excluded.daily_return,
                volume          = excluded.volume,
                volatility_10d  = excluded.volatility_10d,
                mapped_features = excluded.mapped_features,
                last_updated    = excluded.last_updated,
                is_stale        = 0
        """, (
            ticker, bank_name,
            data.get("close_price"),
            data.get("daily_return"),
            data.get("volume"),
            data.get("volatility_10d"),
            mapped_json,
            now,
        ))
        conn.commit()
        conn.close()

    def mark_stale(self, ticker: str) -> None:
        """Marks a ticker as stale (yfinance unavailable)."""
        conn = self._connect()
        conn.execute(
            "UPDATE live_cache SET is_stale = 1 WHERE ticker = ?", (ticker,)
        )
        conn.commit()
        conn.close()

    def is_stale(self, ticker: str) -> bool:
        """Returns True if no data or data is older than STALE_THRESHOLD_SECONDS (24hr)."""
        conn = self._connect()
        row = conn.execute(
            "SELECT last_updated, is_stale FROM live_cache WHERE ticker = ?", (ticker,)
        ).fetchone()
        conn.close()

        if row is None:
            return True
        last_updated_str, is_stale_flag = row
        if is_stale_flag:
            return True
        if not last_updated_str:
            return True
        try:
            last_updated = datetime.fromisoformat(last_updated_str)
            age = (datetime.now() - last_updated).total_seconds()
            return age > STALE_THRESHOLD_SECONDS
        except ValueError:
            return True

    def get_last_update(self, ticker: str) -> Optional[str]:
        """Returns ISO timestamp of last successful fetch, or None."""
        conn = self._connect()
        row = conn.execute(
            "SELECT last_updated FROM live_cache WHERE ticker = ?", (ticker,)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def get_all_tickers(self) -> list:
        """Returns list of all tickers currently in the cache."""
        conn = self._connect()
        rows = conn.execute("SELECT ticker FROM live_cache").fetchall()
        conn.close()
        return [r[0] for r in rows]

    def clear(self) -> None:
        """Clears all cache entries. Used in tests."""
        conn = self._connect()
        conn.execute("DELETE FROM live_cache")
        conn.commit()
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
