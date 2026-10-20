"""
Financial Contagion Monitor v2 — src/data/live_loader.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

LiveDataLoader — fetches real NSE/BSE stock data via yfinance and maps
stock signals to the BankSnapshot format all downstream modules expect.

Key design:
  - Parallel fetch using ThreadPoolExecutor (33 tickers in ~3-4s, not ~30s)
  - 5-minute SQLite cache — all reads go through cache, never direct to yfinance
  - Feature mapping: stock signals → financial feature deltas on top of CSV baseline
  - Returns DayResult with data_mode="live" — identical contract to ReplayEngine

Market hours: NSE 9:15 AM – 3:30 PM IST, Monday–Friday
Outside hours: yfinance returns last closing price — still valid for the model
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

try:
    import yfinance as yf
    import pytz
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from src.data.loader import (
    BankSnapshot, DayResult, RealDataLoader,
    compute_sri, get_top_risks, get_active_crisis,
)
from src.data.cache import LiveCache
from src.ews.detector import EarlyWarningDetector

# ── NSE ticker map: institution_id → NSE ticker ───────────────────────────────
TICKER_MAP = {
    "State Bank of India (SBI)":    "SBIN.NS",
    "HDFC Bank":                    "HDFCBANK.NS",
    "ICICI Bank":                   "ICICIBANK.NS",
    "Axis Bank":                    "AXISBANK.NS",
    "Kotak Mahindra Bank":          "KOTAKBANK.NS",
    "AU Small Finance Bank":        "AUBANK.NS",
    "Bandhan Bank":                 "BANDHANBNK.NS",
    "Bank of Baroda":               "BANKBARODA.NS",
    "Bank of India":                "BANKINDIA.NS",
    "Bank of Maharashtra":          "MAHABANK.NS",
    "Canara Bank":                  "CANBK.NS",
    "Central Bank of India":        "CENTRALBK.NS",
    "City Union Bank":              "CUB.NS",
    "CSB Bank":                     "CSBBANK.NS",
    "Equitas Small Finance Bank":   "EQUITASBNK.NS",
    "Federal Bank":                 "FEDERALBNK.NS",
    "IDBI Bank":                    "IDBI.NS",
    "IDFC FIRST Bank":              "IDFCFIRSTB.NS",
    "Indian Bank":                  "INDIANB.NS",
    "Indian Overseas Bank (IOB)":   "IOB.NS",
    "IndusInd Bank":                "INDUSINDBK.NS",
    "J&K Bank":                     "J-KBANK.NS",
    "Karnataka Bank":               "KTKBANK.NS",
    "Karur Vysya Bank":             "KARURVYSYA.NS",
    "Punjab & Sind Bank":           "PSB.NS",
    "Punjab National Bank (PNB)":   "PNB.NS",
    "RBL Bank":                     "RBLBANK.NS",
    "South Indian Bank":            "SOUTHBANK.NS",
    "Tamilnad Mercantile Bank":     "TMB.NS",
    "UCO Bank":                     "UCOBANK.NS",
    "Ujjivan Small Finance Bank":   "UJJIVANSFB.NS",
    "Union Bank of India":          "UNIONBANK.NS",
    "Yes Bank":                     "YESBANK.NS",
}

# NSE market holidays 2025–2026 (approximate — verify at nseindia.com before demo)
NSE_HOLIDAYS = {
    "2025-01-26", "2025-02-26", "2025-03-14", "2025-04-14",
    "2025-04-18", "2025-05-01", "2025-08-15", "2025-10-02",
    "2025-10-20", "2025-10-21", "2025-11-05", "2025-12-25",
    "2026-01-26", "2026-03-03",
}


class LiveDataLoader:
    """
    Fetches real NSE stock data and returns DayResult objects
    in the same format as ReplayEngine — fully mode-agnostic downstream.

    Usage:
        loader = LiveDataLoader(cache, base_loader, ews)
        result = loader.get_current()
        print(result.sri, result.data_mode)   # "live"
    """

    POLL_INTERVAL = int(os.getenv("YFINANCE_POLL_INTERVAL", "300"))  # 5 min

    def __init__(
        self,
        cache: LiveCache,
        base_loader: RealDataLoader,
        ews: Optional[EarlyWarningDetector] = None,
        gnn_engine=None,
    ):
        self._cache = cache
        self._base_loader = base_loader
        self._ews = ews
        self._gnn_engine = gnn_engine
        self._last_fetch_time: Optional[datetime] = None
        self._last_result: Optional[DayResult] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_current(self) -> DayResult:
        """
        Returns a DayResult with current market data.
        Uses cache if data is fresh (< POLL_INTERVAL seconds old).
        Fetches from yfinance if cache is stale.

        Returns:
            DayResult with data_mode="live"
        """
        now = datetime.now()
        needs_refresh = (
            self._last_fetch_time is None or
            (now - self._last_fetch_time).total_seconds() >= self.POLL_INTERVAL
        )

        if needs_refresh:
            self._refresh_cache()
            self._last_fetch_time = now

        return self._build_day_result()

    def is_market_open(self) -> bool:
        """Returns True if NSE is currently trading."""
        return self._is_market_open()

    def get_market_status_text(self) -> str:
        """Returns formatted market status string for the header badge."""
        now_ist = self._get_ist_now()
        if not YFINANCE_AVAILABLE:
            return "MARKET: DATA UNAVAILABLE"
        if now_ist.weekday() >= 5:
            return f"MARKET: WEEKEND ({now_ist.strftime('%H:%M')} IST)"
        if now_ist.strftime("%Y-%m-%d") in NSE_HOLIDAYS:
            return f"MARKET: HOLIDAY ({now_ist.strftime('%d %b')})"
        if self._is_market_open():
            return f"MARKET: OPEN ● {now_ist.strftime('%H:%M:%S')} IST"
        return f"MARKET: CLOSED ● Last close {now_ist.strftime('%d %b')}"

    # ── Private: Cache Refresh ────────────────────────────────────────────────

    def _refresh_cache(self) -> None:
        """Fetches all 33 tickers in parallel and updates the cache."""
        if not YFINANCE_AVAILABLE:
            return

        ticker_data = self._fetch_all_tickers()
        for bank_name, data in ticker_data.items():
            ticker = TICKER_MAP.get(bank_name)
            if ticker and data:
                self._cache.set(ticker, bank_name, data)

    def _fetch_all_tickers(self) -> Dict[str, dict]:
        """
        Fetches yfinance data for all 33 tickers in parallel.
        Uses ThreadPoolExecutor — sequential would take ~30s, parallel ~3-4s.
        """
        results = {}
        bank_names = list(TICKER_MAP.keys())

        def fetch_one(bank_name: str) -> tuple:
            ticker_sym = TICKER_MAP[bank_name]
            try:
                ticker = yf.Ticker(ticker_sym)
                hist = ticker.history(period="15d")
                if hist.empty:
                    return bank_name, None
                data = self._extract_ticker_data(hist)
                return bank_name, data
            except Exception:
                return bank_name, None

        # Parallel fetch with max 10 workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_one, bn): bn for bn in bank_names}
            for future in as_completed(futures):
                bank_name, data = future.result()
                results[bank_name] = data
                time.sleep(0.02)  # tiny sleep between completions to avoid rate spike

        fetched = sum(1 for v in results.values() if v is not None)
        print(f"[LiveLoader] Fetched {fetched}/33 tickers")
        return results

    def _extract_ticker_data(self, hist) -> dict:
        """
        Extracts relevant signals from yfinance OHLCV history DataFrame.
        Returns dict with keys needed by _map_signals_to_features().
        """
        if len(hist) < 2:
            return {}

        closes = hist["Close"].values.astype(float)
        volumes = hist["Volume"].values.astype(float)

        close_today = float(closes[-1])
        close_prev  = float(closes[-2])
        daily_return = (close_today - close_prev) / close_prev if close_prev > 0 else 0.0

        # 5-day return
        if len(closes) >= 5:
            return_5d = (close_today - closes[-5]) / closes[-5] if closes[-5] > 0 else 0.0
        else:
            return_5d = daily_return

        # 10-day volatility
        if len(closes) >= 10:
            daily_rets = np.diff(closes[-10:]) / closes[-10:-1]
            vol_10d = float(np.std(daily_rets))
        else:
            vol_10d = abs(daily_return)

        # Volume spike: compare today vs 5-day average
        avg_volume = float(np.mean(volumes[-5:])) if len(volumes) >= 5 else float(volumes[-1])
        volume_ratio = float(volumes[-1]) / avg_volume if avg_volume > 0 else 1.0

        # 52-week low check
        at_52w_low = False
        if len(closes) >= 252:
            at_52w_low = close_today <= float(np.min(closes[-252:]))
        elif len(closes) >= 10:
            at_52w_low = close_today <= float(np.min(closes))

        return {
            "close_price":    close_today,
            "daily_return":   daily_return,
            "return_5d":      return_5d,
            "volume":         float(volumes[-1]),
            "volume_ratio":   volume_ratio,
            "volatility_10d": vol_10d,
            "at_52w_low":     at_52w_low,
        }

    # ── Private: Feature Mapping ──────────────────────────────────────────────

    def _map_signals_to_features(
        self,
        ticker_data: dict,
        base_snap: BankSnapshot,
    ) -> BankSnapshot:
        """
        Maps stock signals to BankSnapshot financial features.
        Structural features (interbank ratios, capital ratios) remain at
        their last known CSV baseline. Market-sensitive features are updated.

        Stock Signal → Feature Mapping:
          daily_return         → equity_volatility_30d
          5d return < -8%      → vulnerability_label_injected += 0.30
          volume spike + drop  → cds_spread_bps_boosted (panic proxy)
          daily_return < -3%   → distress_event_injected = 1
          10d volatility       → beta_to_financials
          at 52-week low       → capital_buffer (reduce)
        """
        d = base_snap._asdict()

        daily_ret  = ticker_data.get("daily_return", 0.0)
        return_5d  = ticker_data.get("return_5d", 0.0)
        vol_10d    = ticker_data.get("volatility_10d", 0.0)
        vol_ratio  = ticker_data.get("volume_ratio", 1.0)
        at_52w_low = ticker_data.get("at_52w_low", False)

        # 1. Daily return → equity volatility
        d["equity_volatility_30d"] = float(np.clip(abs(daily_ret) * 10, 0.0, 1.0))

        # 2. 5-day return < -8% → sustained downtrend → higher vulnerability
        if return_5d < -0.08:
            d["vulnerability_label_injected"] = float(
                np.clip(d["vulnerability_label_injected"] + 0.30, 0.0, 1.0)
            )

        # 3. Volume spike + price drop → CDS proxy
        if vol_ratio > 2.0 and daily_ret < -0.02:
            boost = min(vol_ratio * abs(daily_ret) * 500, 800)
            d["cds_spread_bps_boosted"] = float(d["cds_spread_bps"] + boost)
            d["cds_spread_bps"] = float(d["cds_spread_bps"] + boost * 0.5)

        # 4. Sharp single-day drop → distress flag
        if daily_ret < -0.03:
            d["distress_event_injected"] = 1

        # 5. 10-day volatility → systematic risk
        d["beta_to_financials"] = float(np.clip(
            d["beta_to_financials"] + vol_10d * 2.0, 0.0, 3.0
        ))

        # 6. At 52-week low → reduce capital buffer (market distress indicator)
        if at_52w_low:
            d["capital_buffer"] = float(max(0.0, d["capital_buffer"] - 0.02))
            d["vulnerability_label_injected"] = float(
                np.clip(d["vulnerability_label_injected"] + 0.10, 0.0, 1.0)
            )

        return BankSnapshot(**d)

    # ── Private: DayResult Construction ──────────────────────────────────────

    def _build_day_result(self) -> DayResult:
        """Builds a complete DayResult from cached live data."""
        today = datetime.now().strftime("%Y-%m-%d")
        snapshots = self._build_snapshots(today)

        if not snapshots:
            return DayResult(
                date=today, snapshots=[], sri=0.0,
                data_mode="live", ews_level="NORMAL",
            )

        # GNN predictions
        if self._gnn_engine is not None:
            try:
                gnn_probs = self._gnn_engine.predict(snapshots)
                updated = []
                for snap in snapshots:
                    prob = gnn_probs.get(snap.institution_id)
                    if prob is not None:
                        d = snap._asdict()
                        d["vulnerability_label_injected"] = float(prob)
                        updated.append(BankSnapshot(**d))
                    else:
                        updated.append(snap)
                snapshots = updated
            except Exception:
                pass

        sri = compute_sri(snapshots)
        top_risks = get_top_risks(snapshots, n=5)
        crisis = get_active_crisis(today)

        ews_level = "NORMAL"
        if self._ews is not None:
            temp = DayResult(
                date=today, snapshots=snapshots, sri=sri,
                crisis_active=crisis is not None,
                crisis_name=crisis.name if crisis else "",
                data_mode="live",
            )
            ews_level = self._ews.update(temp)

        return DayResult(
            date=today,
            snapshots=snapshots,
            sri=round(sri, 6),
            top_risks=top_risks,
            crisis_active=crisis is not None,
            crisis_name=crisis.name if crisis else "",
            ews_level=ews_level,
            data_mode="live",
        )

    def _build_snapshots(self, today: str) -> List[BankSnapshot]:
        """
        Builds 33 BankSnapshots by combining cached live signals
        with CSV baseline structural features.
        """
        snapshots = []
        for bank_name in TICKER_MAP:
            ticker = TICKER_MAP[bank_name]
            base_snap = self._get_base_snapshot(bank_name, today)
            if base_snap is None:
                continue

            # Try fresh cache first, fall back to stale
            cached = self._cache.get(ticker)
            if cached is None:
                cached = self._cache.get_stale(ticker)
                if cached:
                    d = base_snap._asdict()
                    d["distress_event_injected"] = 0
                    base_snap = BankSnapshot(**d)

            if cached and cached.get("mapped_features"):
                snap = self._map_signals_to_features(cached["mapped_features"], base_snap)
            else:
                snap = base_snap

            snapshots.append(snap)

        return snapshots

    def _get_base_snapshot(self, bank_name: str, date: str) -> Optional[BankSnapshot]:
        """
        Gets the last known structural features from the CSV as baseline.
        Uses the most recent available date in the historical dataset.
        """
        try:
            df = self._base_loader.load_generated()
            bank_df = df[df["institution_id"] == bank_name]
            if bank_df.empty:
                return None
            # Use last available date
            last_row = bank_df.sort_values("date").iloc[-1]
            # Override date to today
            snap_dict = {}
            for col in last_row.index:
                snap_dict[col] = last_row[col]
            snap_dict["date"] = date
            # Use loader's row_to_snapshot logic via BankSnapshot fields
            from src.replay.engine import ReplayEngine
            # Build manually to avoid circular import
            def sf(key, default=0.0):
                try:
                    v = last_row.get(key, default)
                    return default if (v != v) else float(v)
                except Exception:
                    return default

            return BankSnapshot(
                institution_id=bank_name,
                bank_short=str(last_row.get("bank_short", "")),
                bank_type=str(last_row.get("bank_type", "periphery")),
                date=date,
                tier1_capital_ratio=sf("tier1_capital_ratio"),
                total_capital_ratio=sf("total_capital_ratio"),
                leverage_ratio=sf("leverage_ratio"),
                capital_buffer=sf("capital_buffer"),
                rwa_density=sf("rwa_density"),
                lcr=sf("lcr"),
                nsfr=sf("nsfr"),
                short_term_funding_ratio=sf("short_term_funding_ratio"),
                npl_ratio=sf("npl_ratio"),
                loan_loss_provision_ratio=sf("loan_loss_provision_ratio"),
                cds_spread_bps=sf("cds_spread_bps"),
                equity_volatility_30d=sf("equity_volatility_30d"),
                beta_to_financials=sf("beta_to_financials"),
                interbank_borrowing_ratio=sf("interbank_borrowing_ratio"),
                interbank_exposure_ratio=sf("interbank_exposure_ratio"),
                network_degree=sf("network_degree"),
                eigenvector_centrality=sf("eigenvector_centrality"),
                gsib_score=sf("gsib_score"),
                total_assets_usd_bn=sf("total_assets_usd_bn", 1.0),
                roe=sf("roe"),
                cost_to_income=sf("cost_to_income"),
                concentration_hhi=sf("concentration_hhi"),
                vulnerability_label_injected=sf("vulnerability_label_injected"),
                distress_event_injected=int(last_row.get("distress_event_injected", 0)),
                cds_spread_bps_boosted=sf("cds_spread_bps_boosted"),
                equity_volatility_boosted=sf("equity_volatility_boosted"),
            )
        except Exception:
            return None

    # ── Private: Market Hours ─────────────────────────────────────────────────

    def _is_market_open(self) -> bool:
        """Checks if NSE is currently in trading hours."""
        if not YFINANCE_AVAILABLE:
            return False
        now = self._get_ist_now()
        date_str = now.strftime("%Y-%m-%d")
        if now.weekday() >= 5:
            return False
        if date_str in NSE_HOLIDAYS:
            return False
        market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close

    @staticmethod
    def _get_ist_now() -> datetime:
        """Returns current datetime in IST timezone."""
        if YFINANCE_AVAILABLE:
            try:
                import pytz
                ist = pytz.timezone("Asia/Kolkata")
                return datetime.now(ist)
            except Exception:
                pass
        return datetime.now()
