"""
Financial Contagion Monitor v2 — src/data/loader.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

RealDataLoader — the most critical module in the project.
Every other module depends on this producing a correct 1,205,490-row dataframe.

Key responsibilities:
  - Merge two source CSV files into one unified dataframe
  - Construct date column from day/month/year
  - Inject 9 real Indian financial crises with exponential decay severity
  - Add bank metadata (type, short ID)
  - Provide get_daily_snapshot() for ReplayEngine
  - Provide DayResult and BankSnapshot data contracts

PERSON 1 (Kushagra) — implement and test this file first.
CRITICAL: Before running, verify CSV column names match EXPECTED_COLUMNS below.
"""

import os
import json
import warnings
from collections import namedtuple
from dataclasses import dataclass, field
from math import exp
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# CRISIS EVENTS — module-level constant imported by ReplayEngine, EWS, Dashboard
# ══════════════════════════════════════════════════════════════════════════════

CrisisEvent = namedtuple("CrisisEvent", [
    "name",
    "start",         # "YYYY-MM-DD"
    "peak_date",     # "YYYY-MM-DD" — severity is maximum here
    "end",           # "YYYY-MM-DD"
    "severity",      # float [0, 1] — maximum vulnerability injection
    "affected_banks" # list of bank_short strings, or ["ALL"]
])

CRISIS_EVENTS = [
    CrisisEvent(
        "China Yuan Devaluation",
        "2015-08-11", "2015-08-24", "2015-09-15", 0.35,
        ["Axis", "ICICI", "HDFC", "Kotak", "IndusInd"]
    ),
    CrisisEvent(
        "Demonetization",
        "2016-11-08", "2016-11-15", "2016-12-31", 0.60,
        ["ALL"]
    ),
    CrisisEvent(
        "IL&FS Crisis",
        "2018-09-21", "2018-10-01", "2018-11-30", 0.55,
        ["BoB", "BoI", "Canara", "Central", "IDBI", "PNB", "Union"]
    ),
    CrisisEvent(
        "Yes Bank Stress",
        "2019-03-05", "2019-09-01", "2020-03-04", 0.45,
        ["Yes", "RBL", "IndusInd"]
    ),
    CrisisEvent(
        "Yes Bank Moratorium",
        "2020-03-05", "2020-03-05", "2020-03-13", 0.75,
        ["Yes"]
    ),
    CrisisEvent(
        "COVID-19 PANDEMIC",
        "2020-03-15", "2020-03-15", "2020-06-30", 0.92,
        ["ALL"]
    ),
    CrisisEvent(
        "Second COVID Wave",
        "2021-04-01", "2021-04-15", "2021-05-31", 0.50,
        ["ALL"]
    ),
    CrisisEvent(
        "RBI Rate Hike Stress",
        "2022-05-04", "2022-06-08", "2022-09-30", 0.42,
        ["IDFC", "IndusInd", "Kotak", "AU SFB", "Equitas", "Ujjivan"]
    ),
    CrisisEvent(
        "SVB Global Contagion",
        "2023-03-10", "2023-03-13", "2023-04-15", 0.32,
        ["AU SFB", "Equitas", "Ujjivan", "CSB", "CUB"]
    ),
]


def get_active_crisis(date_str: str) -> Optional[CrisisEvent]:
    """Returns the CrisisEvent if the date falls within any crisis window, else None.
    If multiple crises overlap, returns the one with the highest severity."""
    active = [c for c in CRISIS_EVENTS if c.start <= date_str <= c.end]
    if not active:
        return None
    return max(active, key=lambda c: c.severity)


# ══════════════════════════════════════════════════════════════════════════════
# BANK METADATA — mapping from full institution_id to short ID and type
# ══════════════════════════════════════════════════════════════════════════════

# 5 Core D-SIBs (Domestic Systemically Important Banks)
CORE_BANKS = {
    "State Bank of India (SBI)": "SBI",
    "HDFC Bank": "HDFC",
    "ICICI Bank": "ICICI",
    "Axis Bank": "Axis",
    "Kotak Mahindra Bank": "Kotak",
}

# 28 Periphery banks
PERIPHERY_BANKS = {
    "AU Small Finance Bank": "AU SFB",
    "Bandhan Bank": "Bandhan",
    "Bank of Baroda": "BoB",
    "Bank of India": "BoI",
    "Bank of Maharashtra": "BoM",
    "Canara Bank": "Canara",
    "Central Bank of India": "Central",
    "City Union Bank": "CUB",
    "CSB Bank": "CSB",
    "Equitas Small Finance Bank": "Equitas",
    "Federal Bank": "Federal",
    "IDBI Bank": "IDBI",
    "IDFC FIRST Bank": "IDFC",
    "Indian Bank": "Indian",
    "Indian Overseas Bank (IOB)": "IOB",
    "IndusInd Bank": "IndusInd",
    "J&K Bank": "J&K",
    "Karnataka Bank": "KarBank",
    "Karur Vysya Bank": "KVB",
    "Punjab & Sind Bank": "P&S",
    "Punjab National Bank (PNB)": "PNB",
    "RBL Bank": "RBL",
    "South Indian Bank": "SIB",
    "Tamilnad Mercantile Bank": "TMB",
    "UCO Bank": "UCO",
    "Ujjivan Small Finance Bank": "Ujjivan",
    "Union Bank of India": "Union",
    "Yes Bank": "Yes",
}

ALL_BANKS = {**CORE_BANKS, **PERIPHERY_BANKS}

# Reverse mapping: short_id -> full institution_id
SHORT_TO_FULL = {v: k for k, v in ALL_BANKS.items()}

# All 28 expected columns in the source CSV files
EXPECTED_COLUMNS = [
    "institution_id", "day", "month", "year",
    "tier1_capital_ratio", "total_capital_ratio", "leverage_ratio",
    "rwa_density", "capital_buffer", "lcr", "nsfr",
    "short_term_funding_ratio", "interbank_borrowing_ratio",
    "interbank_exposure_ratio", "npl_ratio", "loan_loss_provision_ratio",
    "roe", "cost_to_income", "total_assets_usd_bn",
    "cds_spread_bps", "equity_volatility_30d", "beta_to_financials",
    "gsib_score", "concentration_hhi", "network_degree",
    "eigenvector_centrality", "vulnerability_label", "distress_event",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA CONTRACTS — BankSnapshot and DayResult
# ══════════════════════════════════════════════════════════════════════════════

BankSnapshot = namedtuple("BankSnapshot", [
    # Identity
    "institution_id",          # Full bank name e.g. "State Bank of India (SBI)"
    "bank_short",              # Short ID e.g. "SBI"
    "bank_type",               # "core" or "periphery"
    "date",                    # "YYYY-MM-DD"
    # Capital Adequacy
    "tier1_capital_ratio",     # Core equity / RWA
    "total_capital_ratio",     # Total capital / RWA
    "leverage_ratio",          # Assets / Tier 1 capital
    "capital_buffer",          # Capital above regulatory minimum
    "rwa_density",             # Risk-weighted assets / Total assets
    # Liquidity
    "lcr",                     # Liquidity Coverage Ratio
    "nsfr",                    # Net Stable Funding Ratio
    "short_term_funding_ratio", # Short-term liabilities / Total funding
    # Credit Quality
    "npl_ratio",               # Non-performing loans / Total loans
    "loan_loss_provision_ratio", # Provisions / Total loans
    # Market Risk
    "cds_spread_bps",          # Credit Default Swap spread in basis points
    "equity_volatility_30d",   # 30-day rolling equity volatility
    "beta_to_financials",      # Systematic risk vs financials sector
    # Network / Interbank
    "interbank_borrowing_ratio", # Borrowed from banks / Total liabilities
    "interbank_exposure_ratio",  # Exposed to other banks / Total assets
    "network_degree",            # Normalized degree in interbank network
    "eigenvector_centrality",    # Network influence measure
    "gsib_score",               # Global Systemically Important Bank score
    # Size
    "total_assets_usd_bn",     # Total assets in billion USD
    # Profitability
    "roe",                     # Return on equity
    "cost_to_income",          # Operating cost / Net income
    "concentration_hhi",       # Herfindahl-Hirschman Index
    # Injected Crisis Labels
    "vulnerability_label_injected", # [0,1] crisis-injected vulnerability
    "distress_event_injected",      # {0,1} binary distress flag
    # Boosted Market Signals (during crises)
    "cds_spread_bps_boosted",       # CDS spread amplified during crises
    "equity_volatility_boosted",    # Volatility amplified during crises
])


@dataclass
class DayResult:
    """The universal data contract between all modules.
    Produced by ReplayEngine (historical) and LiveDataLoader (live).
    Consumed by ContagionSimulator, MonteCarloStressTester, GNN,
    EarlyWarningDetector, ReportGenerator, AlertManager, and Dashboard.
    """
    date: str                           # "YYYY-MM-DD"
    snapshots: List[BankSnapshot]       # Always 33 items
    sri: float                          # Systemic Risk Index [0, 1]
    sri_ci_low: float = 0.0            # 95% CI lower bound (from Monte Carlo)
    sri_ci_high: float = 0.0           # 95% CI upper bound (from Monte Carlo)
    top_risks: List[BankSnapshot] = field(default_factory=list)  # Top 5 by vulnerability
    crisis_active: bool = False
    crisis_name: str = ""               # e.g. "COVID-19 PANDEMIC"
    ews_level: str = "NORMAL"          # "NORMAL"/"ELEVATED"/"CRITICAL"
    data_mode: str = "historical"      # "historical" or "live"


# ══════════════════════════════════════════════════════════════════════════════
# REALDATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

class RealDataLoader:
    """
    Loads, merges, validates, and enriches the 1.2M row historical dataset.

    Usage:
        loader = RealDataLoader("data/raw/")
        df = loader.load_from_parts()   # call once — saves banks_daily_data.csv
        snapshots = loader.get_daily_snapshot("2020-03-15")  # 33 BankSnapshots
    """

    OUTPUT_FILENAME = "banks_daily_data.csv"

    def __init__(self, data_dir: str = "data/raw/"):
        self.data_dir = data_dir
        self._df: Optional[pd.DataFrame] = None  # cached after load_from_parts()

    # ── Public API ────────────────────────────────────────────────────────────

    def load_from_parts(self) -> pd.DataFrame:
        """
        Main entry point for main.py.
        1. Reads data_part1_800k.csv and data_part2_405k.csv
        2. Validates columns against EXPECTED_COLUMNS
        3. Merges by row concatenation
        4. Builds date column from day/month/year
        5. Adds bank metadata (bank_type, bank_short)
        6. Injects 9 crisis markers with exponential decay
        7. Saves to banks_daily_data.csv
        8. Caches and returns the dataframe
        """
        output_path = os.path.join(self.data_dir, self.OUTPUT_FILENAME)

        # If already generated, load from cache
        if os.path.exists(output_path):
            print(f"  Found existing {self.OUTPUT_FILENAME} — loading from disk...")
            self._df = pd.read_csv(output_path, low_memory=False)
            print(f"  Loaded {len(self._df):,} rows from cache")
            return self._df

        # Load part 1
        part1_path = os.path.join(self.data_dir, "data_part1_800k.csv")
        print(f"  Loading {part1_path}...")
        df1 = pd.read_csv(part1_path, low_memory=False)
        print(f"  Part 1: {len(df1):,} rows, {len(df1.columns)} columns")

        # Load part 2
        part2_path = os.path.join(self.data_dir, "data_part2_405k.csv")
        print(f"  Loading {part2_path}...")
        df2 = pd.read_csv(part2_path, low_memory=False)
        print(f"  Part 2: {len(df2):,} rows, {len(df2.columns)} columns")

        # Validate columns (check for naming mismatches — critical warning)
        self._validate_columns(df1, "data_part1_800k.csv")
        self._validate_columns(df2, "data_part2_405k.csv")

        # Merge
        print("  Merging parts...")
        df = pd.concat([df1, df2], ignore_index=True)
        print(f"  Merged: {len(df):,} rows")

        # Pipeline
        print("  Building date column...")
        df = self._build_date_column(df)

        print("  Adding bank metadata...")
        df = self._add_bank_metadata(df)

        print("  Injecting 9 crisis markers...")
        df = self._inject_crisis_markers(df)

        # Sort for deterministic replay
        df = df.sort_values(["date", "institution_id"]).reset_index(drop=True)

        # Save
        print(f"  Saving to {output_path}...")
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df):,} rows, {len(df.columns)} columns")

        self._df = df
        return df

    def load_generated(self) -> pd.DataFrame:
        """Load the already-generated banks_daily_data.csv (fast path for dashboard)."""
        if self._df is not None:
            return self._df
        output_path = os.path.join(self.data_dir, self.OUTPUT_FILENAME)
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"{output_path} not found. Run python main.py first."
            )
        self._df = pd.read_csv(output_path, low_memory=False)
        return self._df

    def get_daily_snapshot(self, date: str) -> List[BankSnapshot]:
        """
        Returns a list of 33 BankSnapshot namedtuples for the given date.
        Used by ReplayEngine.seek() on every date change.

        Args:
            date: "YYYY-MM-DD" format

        Returns:
            List of exactly 33 BankSnapshot objects, one per bank.
            If fewer than 33 banks found, logs a warning.
        """
        if self._df is None:
            self.load_generated()

        day_df = self._df[self._df["date"] == date]
        if len(day_df) == 0:
            # Try to find nearest date
            available = sorted(self._df["date"].unique())
            nearest = self._find_nearest_date(date, available)
            if nearest:
                day_df = self._df[self._df["date"] == nearest]
            else:
                return []

        # If multiple rows per bank on same date (intraday slots), take mean
        if len(day_df) > 33:
            day_df = day_df.groupby("institution_id").mean(numeric_only=True).reset_index()
            # Re-attach string columns
            meta = self._df[self._df["date"] == date][
                ["institution_id", "bank_short", "bank_type", "date"]
            ].drop_duplicates("institution_id")
            day_df = day_df.merge(meta, on="institution_id", how="left", suffixes=("", "_meta"))

        snapshots = []
        for _, row in day_df.iterrows():
            snap = self._row_to_snapshot(row, date)
            snapshots.append(snap)

        if len(snapshots) < 33:
            print(f"[WARNING] Only {len(snapshots)} banks found for {date} "
                  f"(expected 33). Some banks may be missing from source data.")

        return snapshots

    def get_sri_series(self, start: str = None, end: str = None) -> pd.DataFrame:
        """
        Returns pre-computed SRI series as a DataFrame with columns [date, sri].
        Reads from sri_series.csv — much faster than computing from 1.2M rows.
        """
        sri_path = os.path.join(self.data_dir, "sri_series.csv")
        if not os.path.exists(sri_path):
            raise FileNotFoundError(
                "sri_series.csv not found. Run python main.py first."
            )
        sri_df = pd.read_csv(sri_path)
        if start:
            sri_df = sri_df[sri_df["date"] >= start]
        if end:
            sri_df = sri_df[sri_df["date"] <= end]
        return sri_df.reset_index(drop=True)

    def get_date_range(self) -> Tuple[str, str]:
        """Returns (min_date, max_date) in the dataset."""
        if self._df is None:
            self.load_generated()
        return self._df["date"].min(), self._df["date"].max()

    def get_available_dates(self) -> List[str]:
        """Returns sorted list of all unique trading dates in the dataset."""
        if self._df is None:
            self.load_generated()
        return sorted(self._df["date"].unique().tolist())

    def get_bank_list(self) -> List[str]:
        """Returns list of all 33 institution_id values."""
        if self._df is None:
            self.load_generated()
        return sorted(self._df["institution_id"].unique().tolist())

    # ── Private: Column Validation ────────────────────────────────────────────

    def _validate_columns(self, df: pd.DataFrame, filename: str) -> None:
        """
        Checks that CSV columns match EXPECTED_COLUMNS.
        Warns on mismatches but does not crash — the project may use
        slightly different column names that still work.
        """
        actual_cols = set(df.columns)
        expected_cols = set(EXPECTED_COLUMNS)

        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        if missing:
            print(f"  [WARNING] {filename} is MISSING expected columns:")
            for col in sorted(missing):
                # Try to suggest similar column
                similar = [c for c in actual_cols if col[:8] in c or c[:8] in col]
                hint = f" (similar: {similar[0]})" if similar else ""
                print(f"           MISSING: {col}{hint}")
            print("  If column names differ in your CSV, update EXPECTED_COLUMNS in loader.py")

        if extra:
            print(f"  [INFO] {filename} has {len(extra)} extra columns "
                  f"(not in EXPECTED_COLUMNS) — these will be kept: "
                  f"{sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")

    # ── Private: Date Construction ────────────────────────────────────────────

    def _build_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a date string column "YYYY-MM-DD" from day, month, year columns.
        Handles invalid dates (e.g. Feb 30) by clamping to last valid day of month.
        """
        # Clamp day to valid range for the given month/year
        def safe_date(row):
            try:
                return pd.Timestamp(
                    year=int(row["year"]),
                    month=int(row["month"]),
                    day=min(int(row["day"]), _days_in_month(int(row["month"]), int(row["year"])))
                ).strftime("%Y-%m-%d")
            except Exception:
                # Fallback: return start of month
                return f"{int(row['year']):04d}-{int(row['month']):02d}-01"

        # Vectorized fast path for clean data
        try:
            df["date"] = pd.to_datetime(
                df[["year", "month", "day"]].rename(columns={
                    "year": "year", "month": "month", "day": "day"
                })
            ).dt.strftime("%Y-%m-%d")
        except Exception:
            # Slow path for messy data
            print("  [INFO] Using slow date construction (some invalid day/month values)")
            df["date"] = df.apply(safe_date, axis=1)

        return df

    # ── Private: Bank Metadata ────────────────────────────────────────────────

    def _add_bank_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds bank_type ("core" or "periphery") and bank_short (e.g. "SBI")
        columns based on institution_id.
        Handles fuzzy matching for slight name variations.
        """
        def get_bank_type(institution_id):
            if institution_id in CORE_BANKS:
                return "core"
            elif institution_id in PERIPHERY_BANKS:
                return "periphery"
            else:
                # Fuzzy fallback
                for name in CORE_BANKS:
                    if name.lower() in institution_id.lower():
                        return "core"
                return "periphery"

        def get_bank_short(institution_id):
            if institution_id in ALL_BANKS:
                return ALL_BANKS[institution_id]
            # Try partial match
            for full_name, short in ALL_BANKS.items():
                if full_name.lower() in institution_id.lower():
                    return short
            return institution_id[:6].upper()  # Fallback

        df["bank_type"] = df["institution_id"].apply(get_bank_type)
        df["bank_short"] = df["institution_id"].apply(get_bank_short)

        type_counts = df.groupby("bank_type")["institution_id"].nunique()
        print(f"  Bank types: core={type_counts.get('core', 0)}, "
              f"periphery={type_counts.get('periphery', 0)}")

        return df

    # ── Private: Crisis Injection ─────────────────────────────────────────────

    def _inject_crisis_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Injects 9 real Indian financial crisis markers using exponential decay.

        For each crisis:
        - Severity is max on the peak_date
        - Decays exponentially: severity * exp(-0.05 * days_from_peak)
        - Only applied to affected banks (or ALL banks if affected_banks=["ALL"])
        - Injected values are stored in NEW columns — original columns untouched

        Special case: COVID-19 (2020-03-15 to 2020-04-15) is FORCED to >= 0.92
        to guarantee the visual spike is unmistakable.

        Output columns added:
          vulnerability_label_injected  — float [0, 1]
          distress_event_injected       — int {0, 1}
          cds_spread_bps_boosted        — float
          equity_volatility_boosted     — float
        """
        # Initialize injected columns from originals
        df["vulnerability_label_injected"] = df["vulnerability_label"].astype(float).clip(0, 1)
        df["distress_event_injected"] = df["distress_event"].astype(int)
        df["cds_spread_bps_boosted"] = df["cds_spread_bps"].astype(float)
        df["equity_volatility_boosted"] = df["equity_volatility_30d"].astype(float)

        # Convert date column to a Series for fast comparison
        date_series = pd.to_datetime(df["date"])

        # CRITICAL: Process crises in explicit chronological order.
        # Yes Bank Stress ends 2020-03-04; Yes Bank Moratorium starts 2020-03-05.
        # If processed out of order, the lower-severity Stress event can
        # overwrite the higher-severity Moratorium injection via np.maximum().
        # Sorting by start date guarantees correct layering.
        # COVID forced override is always applied LAST regardless of sort order.
        sorted_crises = sorted(CRISIS_EVENTS, key=lambda c: c.start)

        for crisis in sorted_crises:
            crisis_start = pd.Timestamp(crisis.start)
            crisis_end = pd.Timestamp(crisis.end)
            crisis_peak = pd.Timestamp(crisis.peak_date)

            # Date mask — rows within crisis window
            date_mask = (date_series >= crisis_start) & (date_series <= crisis_end)

            if not date_mask.any():
                continue

            # Bank mask — which banks are affected
            if crisis.affected_banks == ["ALL"]:
                bank_mask = pd.Series(True, index=df.index)
            else:
                # Match by bank_short
                bank_mask = df["bank_short"].isin(crisis.affected_banks)

            full_mask = date_mask & bank_mask

            if not full_mask.any():
                continue

            # Compute exponential decay from peak date
            affected_dates = date_series[full_mask]
            days_from_peak = (affected_dates - crisis_peak).dt.days.abs()
            decay = (crisis.severity * np.exp(-0.05 * days_from_peak)).clip(0, 1)

            # Apply: take the MAX of existing value and new decay
            # (so overlapping crises don't cancel each other)
            current = df.loc[full_mask, "vulnerability_label_injected"]
            df.loc[full_mask, "vulnerability_label_injected"] = np.maximum(
                current.values, decay.values
            )

            # Set distress flag
            df.loc[full_mask, "distress_event_injected"] = 1

            # Boost market signals during crisis
            # CDS spread: amplify by severity factor
            boost_factor = 1.0 + (crisis.severity * 3.0)  # up to 4x for severe crises
            df.loc[full_mask, "cds_spread_bps_boosted"] = (
                df.loc[full_mask, "cds_spread_bps"] * boost_factor
            ).clip(upper=2000)  # cap at 2000 bps

            # Equity volatility: amplify during crises
            vol_boost = 1.0 + (crisis.severity * 1.5)
            df.loc[full_mask, "equity_volatility_boosted"] = (
                df.loc[full_mask, "equity_volatility_30d"] * vol_boost
            ).clip(upper=1.0)

            n_affected = full_mask.sum()
            print(f"  Injected: {crisis.name} ({crisis.start} to {crisis.end}) "
                  f"— {n_affected:,} rows, severity {crisis.severity:.0%}")

        # ── COVID-19 FORCED OVERRIDE ──────────────────────────────────────────
        # The COVID spike must be visually unmistakable from across a room.
        # All rows between 2020-03-15 and 2020-04-15 are guaranteed >= 0.92.
        covid_mask = (
            (date_series >= pd.Timestamp("2020-03-15")) &
            (date_series <= pd.Timestamp("2020-04-15"))
        )
        if covid_mask.any():
            df.loc[covid_mask, "vulnerability_label_injected"] = (
                df.loc[covid_mask, "vulnerability_label_injected"].clip(lower=0.92)
            )
            print(f"  COVID-19 forced override applied: "
                  f"{covid_mask.sum():,} rows guaranteed >= 0.92")

        # Final clip: ensure all values are in [0, 1]
        df["vulnerability_label_injected"] = (
            df["vulnerability_label_injected"].clip(0, 1)
        )

        return df

    # ── Private: Row → Snapshot ───────────────────────────────────────────────

    def _row_to_snapshot(self, row: pd.Series, date: str) -> BankSnapshot:
        """Converts a DataFrame row to a BankSnapshot namedtuple."""

        def safe_float(key: str, default: float = 0.0) -> float:
            try:
                val = row.get(key, default)
                if pd.isna(val):
                    return default
                return float(val)
            except (TypeError, ValueError):
                return default

        def safe_int(key: str, default: int = 0) -> int:
            try:
                val = row.get(key, default)
                if pd.isna(val):
                    return default
                return int(val)
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

    # ── Private: Utility ──────────────────────────────────────────────────────

    @staticmethod
    def _find_nearest_date(target: str, available: List[str]) -> Optional[str]:
        """
        Finds the nearest available trading date to the target.
        If the target date is a weekend, snaps to the preceding Friday.
        """
        if not available:
            return None
        if target in available:
            return target

        target_ts = pd.Timestamp(target)

        # If weekend, try to find preceding Friday first
        if target_ts.weekday() == 5:  # Saturday
            friday = target_ts - pd.Timedelta(days=1)
            fri_str = friday.strftime("%Y-%m-%d")
            if fri_str in available:
                return fri_str
        elif target_ts.weekday() == 6:  # Sunday
            friday = target_ts - pd.Timedelta(days=2)
            fri_str = friday.strftime("%Y-%m-%d")
            if fri_str in available:
                return fri_str

        # General nearest: find minimum absolute difference
        available_ts = pd.to_datetime(available)
        diffs = abs(available_ts - target_ts)
        nearest_idx = diffs.argmin()
        return available[nearest_idx]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _days_in_month(month: int, year: int) -> int:
    """Returns number of days in a month, accounting for leap years."""
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
        return 28
    return 30  # fallback


def compute_sri(snapshots: List[BankSnapshot]) -> float:
    """
    Computes Systemic Risk Index: weighted average of vulnerability scores.
    Weights = total_assets_usd_bn (larger banks have more systemic impact).

    Used by: ReplayEngine, LiveDataLoader, DayResult construction
    """
    if not snapshots:
        return 0.0
    total_assets = sum(s.total_assets_usd_bn for s in snapshots)
    if total_assets <= 0:
        return float(np.mean([s.vulnerability_label_injected for s in snapshots]))
    weighted_sum = sum(
        s.vulnerability_label_injected * s.total_assets_usd_bn
        for s in snapshots
    )
    return round(weighted_sum / total_assets, 6)


def get_top_risks(snapshots: List[BankSnapshot], n: int = 5) -> List[BankSnapshot]:
    """Returns the top N most vulnerable banks, sorted by vulnerability_label_injected."""
    return sorted(snapshots, key=lambda s: s.vulnerability_label_injected, reverse=True)[:n]
