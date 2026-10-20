"""
Financial Contagion Monitor — tests/conftest.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

Shared pytest fixtures used across all 9 test files.
All fixtures are session-scoped where possible to avoid re-computation.
"""

import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── All 33 bank names ─────────────────────────────────────────────────────────
BANKS_33 = [
    "State Bank of India (SBI)", "HDFC Bank", "ICICI Bank",
    "Axis Bank", "Kotak Mahindra Bank", "AU Small Finance Bank",
    "Bandhan Bank", "Bank of Baroda", "Bank of India",
    "Bank of Maharashtra", "Canara Bank", "Central Bank of India",
    "City Union Bank", "CSB Bank", "Equitas Small Finance Bank",
    "Federal Bank", "IDBI Bank", "IDFC FIRST Bank",
    "Indian Bank", "Indian Overseas Bank (IOB)", "IndusInd Bank",
    "J&K Bank", "Karnataka Bank", "Karur Vysya Bank",
    "Punjab & Sind Bank", "Punjab National Bank (PNB)", "RBL Bank",
    "South Indian Bank", "Tamilnad Mercantile Bank", "UCO Bank",
    "Ujjivan Small Finance Bank", "Union Bank of India", "Yes Bank",
]

CORE_BANKS = [
    "State Bank of India (SBI)", "HDFC Bank", "ICICI Bank",
    "Axis Bank", "Kotak Mahindra Bank",
]


def _make_bank_row(bank_name: str, date: str = "2020-03-15",
                   vuln: float = 0.5, distress: int = 0) -> dict:
    """Creates a single bank row dict with sensible defaults."""
    is_yes = "Yes Bank" in bank_name
    return {
        "institution_id":           bank_name,
        "day": 15, "month": 3, "year": 2020,
        "date":                     date,
        "bank_type":                "core" if bank_name in CORE_BANKS else "periphery",
        "bank_short":               bank_name.split()[0][:6].upper(),
        "tier1_capital_ratio":      0.08 if is_yes else 0.14,
        "total_capital_ratio":      0.12 if is_yes else 0.16,
        "leverage_ratio":           12.0,
        "rwa_density":              0.65,
        "capital_buffer":           0.02 if is_yes else 0.08,
        "lcr":                      0.85 if is_yes else 1.20,
        "nsfr":                     0.90 if is_yes else 1.10,
        "short_term_funding_ratio": 0.35,
        "npl_ratio":                0.28 if is_yes else 0.06,
        "loan_loss_provision_ratio":0.15 if is_yes else 0.04,
        "roe":                      0.02 if is_yes else 0.14,
        "cost_to_income":           0.75,
        "total_assets_usd_bn":      30.0 if bank_name in CORE_BANKS else 5.0,
        "cds_spread_bps":           450.0 if is_yes else 120.0,
        "equity_volatility_30d":    0.45 if is_yes else 0.15,
        "beta_to_financials":       1.4 if is_yes else 0.9,
        "gsib_score":               0.3 if bank_name in CORE_BANKS else 0.05,
        "concentration_hhi":        0.12,
        "network_degree":           0.8 if bank_name in CORE_BANKS else 0.2,
        "eigenvector_centrality":   0.7 if bank_name in CORE_BANKS else 0.15,
        "interbank_borrowing_ratio":0.25 if is_yes else 0.12,
        "interbank_exposure_ratio": 0.18 if is_yes else 0.10,
        "vulnerability_label":      1 if is_yes else 0,
        "distress_event":           distress,
        "vulnerability_label_injected": vuln,
        "distress_event_injected":  distress,
        "cds_spread_bps_boosted":   450.0 if is_yes else 120.0,
        "equity_volatility_boosted":0.45 if is_yes else 0.15,
    }


@pytest.fixture(scope="session")
def sample_df_one_date():
    """33-row DataFrame for 2020-03-15 — fast, avoids loading 1.2M rows."""
    rows = []
    for bank in BANKS_33:
        is_yes  = "Yes Bank" in bank
        is_core = bank in CORE_BANKS
        vuln    = 0.94 if is_yes else (0.92 if is_core else 0.55)
        rows.append(_make_bank_row(bank, "2020-03-15", vuln=vuln,
                                   distress=1 if is_yes else 0))
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def sample_snapshots(sample_df_one_date):
    """33 BankSnapshot namedtuples for 2020-03-15."""
    from src.data.loader import BankSnapshot
    snaps = []
    for _, row in sample_df_one_date.iterrows():
        d = row.to_dict()
        snap = BankSnapshot(
            institution_id=d["institution_id"],
            bank_short=d["bank_short"],
            bank_type=d["bank_type"],
            date=d["date"],
            tier1_capital_ratio=d["tier1_capital_ratio"],
            total_capital_ratio=d["total_capital_ratio"],
            leverage_ratio=d["leverage_ratio"],
            capital_buffer=d["capital_buffer"],
            rwa_density=d["rwa_density"],
            lcr=d["lcr"],
            nsfr=d["nsfr"],
            short_term_funding_ratio=d["short_term_funding_ratio"],
            npl_ratio=d["npl_ratio"],
            loan_loss_provision_ratio=d["loan_loss_provision_ratio"],
            cds_spread_bps=d["cds_spread_bps"],
            equity_volatility_30d=d["equity_volatility_30d"],
            beta_to_financials=d["beta_to_financials"],
            interbank_borrowing_ratio=d["interbank_borrowing_ratio"],
            interbank_exposure_ratio=d["interbank_exposure_ratio"],
            network_degree=d["network_degree"],
            eigenvector_centrality=d["eigenvector_centrality"],
            gsib_score=d["gsib_score"],
            total_assets_usd_bn=d["total_assets_usd_bn"],
            roe=d["roe"],
            cost_to_income=d["cost_to_income"],
            concentration_hhi=d["concentration_hhi"],
            vulnerability_label_injected=d["vulnerability_label_injected"],
            distress_event_injected=d["distress_event_injected"],
            cds_spread_bps_boosted=d["cds_spread_bps_boosted"],
            equity_volatility_boosted=d["equity_volatility_boosted"],
        )
        snaps.append(snap)
    return snaps


@pytest.fixture(scope="session")
def sample_day_result(sample_snapshots):
    """DayResult for 2020-03-15 (COVID peak)."""
    from src.data.loader import DayResult, compute_sri, get_top_risks
    sri = compute_sri(sample_snapshots)
    return DayResult(
        date="2020-03-15",
        snapshots=sample_snapshots,
        sri=sri,
        sri_ci_low=sri - 0.05,
        sri_ci_high=sri + 0.05,
        top_risks=get_top_risks(sample_snapshots, 5),
        crisis_active=True,
        crisis_name="COVID-19 PANDEMIC",
        ews_level="CRITICAL",
        data_mode="historical",
    )


@pytest.fixture(scope="session")
def sample_graph():
    """Minimal PyG graph — 33 nodes, 14 features, ~74 edges."""
    try:
        import torch
        from torch_geometric.data import Data
        x          = torch.randn(33, 14)
        # Core-core edges (bidirectional)
        src, dst = [], []
        for i in range(5):
            for j in range(i+1, 5):
                src += [i, j]
                dst += [j, i]
        # Periphery → first 2 core banks
        for p in range(5, 33):
            src += [p, 0, p, 1]
            dst += [0, p, 1, p]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.rand(len(src), 2)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=33)
    except ImportError:
        return None


@pytest.fixture(scope="session")
def node_mapping_df():
    """node_mapping.csv as DataFrame."""
    rows = [{"bank_name": b, "matrix_index": i} for i, b in enumerate(BANKS_33)]
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def mock_yfinance():
    """Mock yfinance Ticker — returns 15 days of OHLCV data."""
    with patch("yfinance.Ticker") as mock_ticker:
        closes  = np.linspace(100, 95, 15).tolist()
        volumes = [1_000_000] * 15
        hist_df = pd.DataFrame({
            "Close":  closes,
            "Volume": volumes,
            "Open":   closes,
            "High":   [c * 1.01 for c in closes],
            "Low":    [c * 0.99 for c in closes],
        })
        mock_ticker.return_value.history.return_value = hist_df
        yield mock_ticker


@pytest.fixture
def mock_ollama_available():
    with patch("src.reports.llm_client.OllamaClient.is_available", return_value=True):
        with patch("src.reports.llm_client.OllamaClient.generate",
                   return_value="1. Crisis Executive Summary\nTest report.\n"
                                "2. Systemic Risk Index Analysis\nSRI is high.\n"
                                "3. Contagion Chain Analysis\nYes Bank originated.\n"
                                "4. Banks Likely to Be Infected Next\nRBL, IndusInd.\n"
                                "5. RBI Action Required?\nYES.\n"
                                "6. Recovery Timeline Estimate\n45 days."):
            yield


@pytest.fixture
def mock_ollama_unavailable():
    with patch("src.reports.llm_client.OllamaClient.is_available", return_value=False):
        yield
