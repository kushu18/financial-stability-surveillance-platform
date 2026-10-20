"""tests/test_simulation.py — 10 tests for ContagionSimulator."""
import pytest
import numpy as np

from src.simulation.contagion import ContagionSimulator, CORE_BANKS, CascadeResult


def test_cascade_result_has_steps(sample_snapshots):
    """Simulation produces at least 1 step."""
    sim    = ContagionSimulator()
    result = sim.run(sample_snapshots, "Yes Bank", magnitude=0.85,
                     shock_type="Liquidity Crisis", n_steps=3)
    assert len(result.steps) >= 1


def test_origin_is_step_zero(sample_snapshots):
    """Step 0 is always the origin bank."""
    sim    = ContagionSimulator()
    result = sim.run(sample_snapshots, "Yes Bank", magnitude=0.85)
    origin_steps = [s for s in result.steps if s.step == 0]
    assert len(origin_steps) == 1
    assert origin_steps[0].bank_name == "Yes Bank"


def test_yes_bank_cascades_to_neighbors(sample_snapshots):
    """Yes Bank shock propagates beyond step 0."""
    sim    = ContagionSimulator()
    result = sim.run(sample_snapshots, "Yes Bank", magnitude=0.85,
                     shock_type="Liquidity Crisis", n_steps=3)
    assert result.total_banks_affected >= 2, \
        "Yes Bank shock should affect at least 2 banks"


def test_dsib_flag_fires_for_core(sample_snapshots):
    """Shocking a core bank triggers D-SIB flag."""
    sim    = ContagionSimulator()
    result = sim.run(sample_snapshots, "State Bank of India (SBI)",
                     magnitude=0.90, shock_type="Capital Shortfall", n_steps=3)
    # SBI is itself a D-SIB — must be flagged
    assert result.rbi_intervention_required


def test_injection_estimate_positive_when_dsib(sample_snapshots):
    """Estimated injection is positive when a D-SIB is affected."""
    sim    = ContagionSimulator()
    result = sim.run(sample_snapshots, "State Bank of India (SBI)",
                     magnitude=0.90, n_steps=3)
    if result.rbi_intervention_required:
        assert result.estimated_injection_cr > 0


def test_all_shock_types_run(sample_snapshots):
    """All 4 shock types complete without error."""
    sim = ContagionSimulator()
    for shock_type in ["Liquidity Crisis", "Capital Shortfall",
                       "NPL Spike", "Market Panic"]:
        result = sim.run(sample_snapshots, "Yes Bank",
                         magnitude=0.75, shock_type=shock_type, n_steps=2)
        assert isinstance(result, CascadeResult)


def test_magnitude_clipped(sample_snapshots):
    """Magnitude > 1.0 is clipped to 1.0."""
    sim    = ContagionSimulator()
    result = sim.run(sample_snapshots, "Yes Bank", magnitude=1.5)
    assert result.magnitude <= 1.0


def test_to_table_data_format(sample_snapshots):
    """to_table_data returns list of dicts with expected keys."""
    sim    = ContagionSimulator()
    result = sim.run(sample_snapshots, "Yes Bank", magnitude=0.75)
    table  = sim.to_table_data(result)
    assert isinstance(table, list)
    assert len(table) >= 1
    assert "Step"          in table[0]
    assert "Bank"          in table[0]
    assert "Vulnerability" in table[0]


def test_last_result_stored(sample_snapshots):
    """ContagionSimulator stores last_result after run."""
    sim = ContagionSimulator()
    sim.run(sample_snapshots, "Yes Bank", magnitude=0.75)
    assert sim.last_result is not None


def test_zero_magnitude_minimal_cascade(sample_snapshots):
    """Very small magnitude produces minimal cascade."""
    sim    = ContagionSimulator()
    result = sim.run(sample_snapshots, "Yes Bank", magnitude=0.05, n_steps=3)
    # With 5% magnitude, few banks should cascade beyond origin
    assert result.total_banks_affected <= 5
