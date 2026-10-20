"""
Financial Contagion Monitor — src/replay/engine.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

ReplayEngine — loads the full 1.2M row dataset into memory and provides
seek/step operations for the historical dashboard replay mode.

In Historical Mode:  primary data source for every date change
In Live Mode:        used only for the contradiction panel's historical benchmark

Key methods:
  seek(date)         — jump to any date, snap to nearest if weekend
  step(direction)    — advance or rewind one trading day
  jump_to_crisis()   — jump directly to a crisis peak date
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.loader import (
    BankSnapshot, DayResult, RealDataLoader,
    CRISIS_EVENTS, get_active_crisis,
    compute_sri, get_top_risks,
)


class ReplayEngine:
    """
    Stateful replay engine over the 3,653-day historical dataset.

    Usage:
        engine = ReplayEngine(loader)
        engine.load()
        result = engine.seek("2020-03-15")
        print(result.sri)          # ~0.92
        print(result.crisis_name)  # "COVID-19 PANDEMIC"
    """

    def __init__(
        self,
        loader: RealDataLoader,
        ews=None,          # EarlyWarningDetector (injected to avoid circular import)
        gnn_engine=None,   # GNNInferenceEngine (injected)
        mc=None,           # MonteCarloStressTester (injected, optional)
    ):
        self._loader = loader
        self._ews = ews
        self._gnn_engine = gnn_engine
        self._mc = mc

        # Full dataset — loaded once, kept in memory (~550MB)
        self._df: Optional[pd.DataFrame] = None

        # Index: date_str -> list of row indices for fast seek
        self._date_index: Dict[str, List[int]] = {}

        # Ordered list of all available trading dates
        self._available_dates: List[str] = []
        self._current_date_idx: int = 0  # pointer into _available_dates

        # Pre-loaded SRI series (3653 rows, read from sri_series.csv)
        self._sri_df: Optional[pd.DataFrame] = None

        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Loads the full dataset into memory.
        Called once at dashboard startup.
        ~550MB RAM usage — acceptable on 8GB+ systems.
        """
        print("[ReplayEngine] Loading dataset into memory...")
        self._df = self._loader.load_generated()

        # Build date index
        print("[ReplayEngine] Building date index...")
        for idx, row in self._df.iterrows():
            date_str = str(row["date"])
            if date_str not in self._date_index:
                self._date_index[date_str] = []
            self._date_index[date_str].append(idx)

        self._available_dates = sorted(self._date_index.keys())
        n_dates = len(self._available_dates)
        print(f"[ReplayEngine] Indexed {n_dates} trading dates "
              f"({self._available_dates[0]} to {self._available_dates[-1]})")

        # Load pre-computed SRI series
        try:
            self._sri_df = self._loader.get_sri_series()
            print(f"[ReplayEngine] SRI series loaded: {len(self._sri_df)} rows")
        except FileNotFoundError:
            print("[ReplayEngine] sri_series.csv not found — SRI chart will be empty")

        self._loaded = True

        # Set initial position to COVID peak (most dramatic opening state for demo)
        covid_date = "2020-03-15"
        if covid_date in self._date_index:
            self._current_date_idx = self._available_dates.index(covid_date)

    def seek(self, date: str) -> DayResult:
        """
        Jumps to the specified date and returns a DayResult.

        If the date is a weekend or not in the dataset, snaps to the
        nearest available trading date and notes this in the result.

        Args:
            date: "YYYY-MM-DD"

        Returns:
            DayResult for the (possibly snapped) date
        """
        if not self._loaded:
            raise RuntimeError("Call load() before seek().")

        # Find the actual date to use (snap if needed)
        actual_date = self._resolve_date(date)
        if actual_date in self._available_dates:
            self._current_date_idx = self._available_dates.index(actual_date)

        return self._build_day_result(actual_date)

    def step(self, direction: int = 1) -> DayResult:
        """
        Moves one trading day forward (direction=1) or backward (direction=-1).

        Returns:
            DayResult for the new date
        """
        if not self._loaded:
            raise RuntimeError("Call load() before step().")

        new_idx = self._current_date_idx + direction
        new_idx = max(0, min(new_idx, len(self._available_dates) - 1))
        self._current_date_idx = new_idx
        date = self._available_dates[new_idx]
        return self._build_day_result(date)

    def jump_to_crisis(self, crisis_name: str) -> DayResult:
        """
        Jumps directly to the peak date of the named crisis.

        Args:
            crisis_name: Must match CrisisEvent.name exactly
                         (e.g. "COVID-19 PANDEMIC", "Demonetization")
        Returns:
            DayResult at the crisis peak date
        """
        for crisis in CRISIS_EVENTS:
            if crisis.name.lower() == crisis_name.lower():
                return self.seek(crisis.peak_date)
        raise ValueError(
            f"Crisis '{crisis_name}' not found. "
            f"Available: {[c.name for c in CRISIS_EVENTS]}"
        )

    def get_available_dates(self) -> List[str]:
        """Returns all available trading dates in sorted order."""
        if not self._loaded:
            raise RuntimeError("Call load() before get_available_dates().")
        return self._available_dates.copy()

    def get_current_date(self) -> str:
        """Returns the currently selected date."""
        if not self._available_dates:
            return ""
        return self._available_dates[self._current_date_idx]

    def get_sri_series_for_chart(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Returns the pre-computed SRI series for the chart.
        Reads from self._sri_df (loaded once at startup).
        O(1) — never scans the 1.2M row dataset.

        Returns:
            DataFrame with columns [date, sri]
        """
        if self._sri_df is None:
            return pd.DataFrame(columns=["date", "sri"])
        df = self._sri_df.copy()
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]
        return df.reset_index(drop=True)

    def get_min_date(self) -> str:
        if not self._available_dates:
            return "2015-01-01"
        return self._available_dates[0]

    def get_max_date(self) -> str:
        if not self._available_dates:
            return "2024-12-31"
        return self._available_dates[-1]

    # ── Private: DayResult Construction ──────────────────────────────────────

    def _build_day_result(self, date: str) -> DayResult:
        """
        Builds a complete DayResult for the given date.
        Aggregates all rows for that date into 33 BankSnapshots.
        Computes SRI, detects active crises, queries EWS.
        """
        # Get 33 snapshots
        snapshots = self._get_snapshots_for_date(date)

        if not snapshots:
            # Fallback: return empty result
            return DayResult(
                date=date,
                snapshots=[],
                sri=0.0,
                crisis_active=False,
                crisis_name="",
                ews_level="NORMAL",
                data_mode="historical",
            )

        # Compute SRI
        sri = self._compute_sri(snapshots)

        # Compute SRI CI from Monte Carlo if available
        sri_ci_low, sri_ci_high = sri, sri
        if self._mc is not None:
            mc_result = self._mc.get_last_result()
            if mc_result is not None:
                sri_ci_low = mc_result.sri_ci_low
                sri_ci_high = mc_result.sri_ci_high

        # Get GNN vulnerability probabilities (if engine available)
        if self._gnn_engine is not None:
            try:
                gnn_probs = self._gnn_engine.predict(snapshots)
                # Update snapshot vulnerability scores with GNN predictions
                snapshots = self._merge_gnn_probs(snapshots, gnn_probs)
            except Exception:
                pass  # fall through — use raw vulnerability_label_injected

        # Top 5 at-risk banks
        top_risks = get_top_risks(snapshots, n=5)

        # Crisis detection
        crisis = get_active_crisis(date)
        crisis_active = crisis is not None
        crisis_name = crisis.name if crisis else ""

        # EWS level
        ews_level = "NORMAL"
        if self._ews is not None:
            day_result_temp = DayResult(
                date=date, snapshots=snapshots, sri=sri,
                crisis_active=crisis_active, crisis_name=crisis_name,
                data_mode="historical",
            )
            ews_level = self._ews.update(day_result_temp)

        return DayResult(
            date=date,
            snapshots=snapshots,
            sri=round(sri, 6),
            sri_ci_low=round(sri_ci_low, 6),
            sri_ci_high=round(sri_ci_high, 6),
            top_risks=top_risks,
            crisis_active=crisis_active,
            crisis_name=crisis_name,
            ews_level=ews_level,
            data_mode="historical",
        )

    def _get_snapshots_for_date(self, date: str) -> List[BankSnapshot]:
        """
        Returns 33 BankSnapshots for a date.
        If multiple rows per bank (intraday slots), takes the mean.
        """
        if date not in self._date_index:
            return []

        row_indices = self._date_index[date]
        day_df = self._df.iloc[row_indices]

        # If multiple rows per bank, group by institution_id and average
        if len(day_df) > 33:
            # Average numeric columns, keep first for string columns
            num_cols = day_df.select_dtypes(include=np.number).columns.tolist()
            str_cols = ["institution_id", "bank_short", "bank_type", "date"]

            avg_df = day_df.groupby("institution_id")[num_cols].mean().reset_index()
            str_df = day_df.groupby("institution_id")[
                [c for c in str_cols if c != "institution_id"]
            ].first().reset_index()

            day_df = avg_df.merge(str_df, on="institution_id", how="left")

        # Convert rows to BankSnapshot namedtuples
        snapshots = []
        for _, row in day_df.iterrows():
            try:
                snap = self._row_to_snapshot(row, date)
                snapshots.append(snap)
            except Exception as e:
                print(f"[ReplayEngine] Skipping row for {row.get('institution_id', '?')}: {e}")

        return snapshots

    def _row_to_snapshot(self, row: pd.Series, date: str) -> BankSnapshot:
        """Converts a DataFrame row to BankSnapshot."""
        def safe_float(key, default=0.0):
            try:
                v = row.get(key, default)
                return default if pd.isna(v) else float(v)
            except (TypeError, ValueError):
                return default

        def safe_int(key, default=0):
            try:
                v = row.get(key, default)
                return default if pd.isna(v) else int(v)
            except (TypeError, ValueError):
                return default

        return BankSnapshot(
            institution_id=str(row.get("institution_id", "")),
            bank_short=str(row.get("bank_short", "")),
            bank_type=str(row.get("bank_type", "periphery")),
            date=str(row.get("date", date)),
            tier1_capital_ratio=safe_float("tier1_capital_ratio"),
            total_capital_ratio=safe_float("total_capital_ratio"),
            leverage_ratio=safe_float("leverage_ratio"),
            capital_buffer=safe_float("capital_buffer"),
            rwa_density=safe_float("rwa_density"),
            lcr=safe_float("lcr"),
            nsfr=safe_float("nsfr"),
            short_term_funding_ratio=safe_float("short_term_funding_ratio"),
            npl_ratio=safe_float("npl_ratio"),
            loan_loss_provision_ratio=safe_float("loan_loss_provision_ratio"),
            cds_spread_bps=safe_float("cds_spread_bps"),
            equity_volatility_30d=safe_float("equity_volatility_30d"),
            beta_to_financials=safe_float("beta_to_financials"),
            interbank_borrowing_ratio=safe_float("interbank_borrowing_ratio"),
            interbank_exposure_ratio=safe_float("interbank_exposure_ratio"),
            network_degree=safe_float("network_degree"),
            eigenvector_centrality=safe_float("eigenvector_centrality"),
            gsib_score=safe_float("gsib_score"),
            total_assets_usd_bn=safe_float("total_assets_usd_bn", 1.0),
            roe=safe_float("roe"),
            cost_to_income=safe_float("cost_to_income"),
            concentration_hhi=safe_float("concentration_hhi"),
            vulnerability_label_injected=safe_float("vulnerability_label_injected"),
            distress_event_injected=safe_int("distress_event_injected"),
            cds_spread_bps_boosted=safe_float("cds_spread_bps_boosted"),
            equity_volatility_boosted=safe_float("equity_volatility_boosted"),
        )

    def _compute_sri(self, snapshots: List[BankSnapshot]) -> float:
        """
        Systemic Risk Index: asset-weighted average vulnerability.
        Larger banks have more systemic impact.
        """
        return compute_sri(snapshots)

    def _merge_gnn_probs(
        self,
        snapshots: List[BankSnapshot],
        gnn_probs: Dict[str, float],
    ) -> List[BankSnapshot]:
        """
        Updates snapshots with GNN vulnerability probabilities.
        Replaces vulnerability_label_injected with GNN P(vulnerable).
        """
        merged = []
        for snap in snapshots:
            gnn_vuln = gnn_probs.get(snap.institution_id)
            if gnn_vuln is not None:
                d = snap._asdict()
                d["vulnerability_label_injected"] = float(gnn_vuln)
                merged.append(BankSnapshot(**d))
            else:
                merged.append(snap)
        return merged

    # ── Private: Date Resolution ──────────────────────────────────────────────

    def _resolve_date(self, date: str) -> str:
        """
        Resolves a requested date to an available trading date.
        Weekend → snap to nearest Friday.
        Not in index → snap to nearest available date.
        """
        if date in self._date_index:
            return date

        if not self._available_dates:
            return date

        target = pd.Timestamp(date)

        # Try preceding Friday for weekends
        if target.weekday() == 5:  # Saturday
            fri = (target - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if fri in self._date_index:
                return fri
        elif target.weekday() == 6:  # Sunday
            fri = (target - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            if fri in self._date_index:
                return fri

        # General nearest
        available_ts = pd.to_datetime(self._available_dates)
        diffs = abs(available_ts - target)
        return self._available_dates[diffs.argmin()]
