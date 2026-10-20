"""tests/test_loader.py — 20 tests for RealDataLoader, crisis injection, data contracts."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.loader import (
    RealDataLoader, BankSnapshot, DayResult,
    CRISIS_EVENTS, get_active_crisis, compute_sri, get_top_risks,
    ALL_BANKS, CORE_BANKS, PERIPHERY_BANKS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def minimal_df():
    """33-row DataFrame — one date, all banks. No file I/O."""
    from tests.conftest import BANKS_33, _make_bank_row
    rows = []
    for bank in BANKS_33:
        is_yes  = "Yes Bank" in bank
        vuln    = 0.94 if is_yes else 0.50
        rows.append(_make_bank_row(bank, "2020-03-15", vuln=vuln,
                                   distress=1 if is_yes else 0))
    return pd.DataFrame(rows)


# ── Section 1: Data contract tests ───────────────────────────────────────────

def test_bank_snapshot_fields():
    """BankSnapshot has all expected fields."""
    fields = BankSnapshot._fields
    assert "institution_id"           in fields
    assert "vulnerability_label_injected" in fields
    assert "distress_event_injected"  in fields
    assert "npl_ratio"                in fields
    assert "tier1_capital_ratio"      in fields
    assert "total_assets_usd_bn"      in fields
    assert "eigenvector_centrality"   in fields


def test_day_result_fields():
    """DayResult dataclass has all expected fields."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(DayResult)}
    assert "date"           in field_names
    assert "snapshots"      in field_names
    assert "sri"            in field_names
    assert "crisis_active"  in field_names
    assert "ews_level"      in field_names
    assert "data_mode"      in field_names
    assert "top_risks"      in field_names


def test_all_banks_count():
    """ALL_BANKS dict has exactly 33 entries."""
    assert len(ALL_BANKS) == 33


def test_core_periphery_split():
    """5 core banks, 28 periphery banks."""
    assert len(CORE_BANKS) == 5
    assert len(PERIPHERY_BANKS) == 28
    assert len(set(CORE_BANKS) & set(PERIPHERY_BANKS)) == 0


# ── Section 2: Crisis events ──────────────────────────────────────────────────

def test_crisis_events_count():
    """Exactly 9 crisis events are defined."""
    assert len(CRISIS_EVENTS) == 9


def test_crisis_names_unique():
    """All crisis names are unique."""
    names = [c.name for c in CRISIS_EVENTS]
    assert len(names) == len(set(names))


def test_covid_crisis_severity():
    """COVID-19 crisis has severity >= 0.90."""
    covid = next(c for c in CRISIS_EVENTS if "COVID-19" in c.name)
    assert covid.severity >= 0.90


def test_get_active_crisis_covid():
    """COVID-19 crisis is active on 2020-03-15."""
    crisis = get_active_crisis("2020-03-15")
    assert crisis is not None
    assert "COVID" in crisis.name


def test_get_active_crisis_none():
    """No crisis is active on a normal date."""
    crisis = get_active_crisis("2017-06-15")
    assert crisis is None


def test_crisis_chronological_order():
    """CRISIS_EVENTS sorted by start date produces chronological order."""
    sorted_crises = sorted(CRISIS_EVENTS, key=lambda c: c.start)
    starts = [c.start for c in sorted_crises]
    assert starts == sorted(starts)


# ── Section 3: Injection logic (no file I/O) ──────────────────────────────────

def test_covid_forced_override(minimal_df):
    """vulnerability_label_injected >= 0.92 for all banks on 2020-03-15."""
    loader = RealDataLoader.__new__(RealDataLoader)
    loader.data_dir = "data/raw/"
    loader._df = None
    df = loader._inject_crisis_markers(minimal_df.copy())
    covid_rows = df[df["date"] == "2020-03-15"]
    assert len(covid_rows) > 0
    assert (covid_rows["vulnerability_label_injected"] >= 0.92).all(), \
        f"Min COVID vuln = {covid_rows['vulnerability_label_injected'].min():.4f}"


def test_yes_bank_distress_during_moratorium(minimal_df):
    """Yes Bank has distress_event_injected=1 in its moratorium window."""
    loader = RealDataLoader.__new__(RealDataLoader)
    loader.data_dir = "data/raw/"
    loader._df = None

    # Add moratorium date rows with correct bank_short for crisis matching
    extra_rows = []
    for date in ["2020-03-05", "2020-03-10"]:
        from tests.conftest import _make_bank_row
        row = _make_bank_row("Yes Bank", date, vuln=0.3, distress=0)
        row["bank_short"] = "Yes"  # Must match PERIPHERY_BANKS["Yes Bank"] = "Yes"
        extra_rows.append(row)
    df = pd.concat([minimal_df, pd.DataFrame(extra_rows)], ignore_index=True)
    # Fix bank_short for all Yes Bank rows to match crisis affected_banks
    df.loc[df["institution_id"] == "Yes Bank", "bank_short"] = "Yes"
    df = loader._inject_crisis_markers(df)

    yes_moratorium = df[
        (df["institution_id"] == "Yes Bank") &
        (df["date"] >= "2020-03-05") &
        (df["date"] <= "2020-03-13")
    ]
    assert (yes_moratorium["distress_event_injected"] == 1).all()


def test_injection_preserves_original_columns(minimal_df):
    """Original vulnerability_label column is not modified by injection."""
    original_values = minimal_df["vulnerability_label"].copy()
    loader = RealDataLoader.__new__(RealDataLoader)
    loader.data_dir = "data/raw/"
    loader._df = None
    df = loader._inject_crisis_markers(minimal_df.copy())
    pd.testing.assert_series_equal(df["vulnerability_label"], original_values)


def test_injection_creates_new_columns(minimal_df):
    """Injection adds all 4 new columns."""
    loader = RealDataLoader.__new__(RealDataLoader)
    loader.data_dir = "data/raw/"
    loader._df = None
    df = loader._inject_crisis_markers(minimal_df.copy())
    assert "vulnerability_label_injected" in df.columns
    assert "distress_event_injected"      in df.columns
    assert "cds_spread_bps_boosted"       in df.columns
    assert "equity_volatility_boosted"    in df.columns


# ── Section 4: SRI computation ────────────────────────────────────────────────

def test_compute_sri_range(sample_snapshots):
    """SRI is always in [0, 1]."""
    sri = compute_sri(sample_snapshots)
    assert 0.0 <= sri <= 1.0


def test_compute_sri_covid_high(sample_snapshots):
    """SRI > 0.70 for COVID peak snapshots (weighted by total_assets_usd_bn)."""
    sri = compute_sri(sample_snapshots)
    assert sri > 0.70, f"Expected SRI > 0.70 for COVID peak, got {sri:.4f}"


def test_compute_sri_empty():
    """SRI is 0.0 for empty snapshot list."""
    assert compute_sri([]) == 0.0


def test_get_top_risks_count(sample_snapshots):
    """get_top_risks returns exactly N banks."""
    top = get_top_risks(sample_snapshots, n=5)
    assert len(top) == 5


def test_yes_bank_in_top_risks(sample_snapshots):
    """Yes Bank is always in the top risks for COVID date."""
    top = get_top_risks(sample_snapshots, n=5)
    bank_names = [s.institution_id for s in top]
    assert "Yes Bank" in bank_names
