"""tests/test_montecarlo.py — 10 tests for MonteCarloStressTester."""
import pytest
import numpy as np

from src.simulation.montecarlo import MonteCarloStressTester, MCResult


def test_mc_runs_sync(sample_snapshots):
    """Synchronous MC run completes and returns MCResult."""
    mc     = MonteCarloStressTester(n_runs=50)
    result = mc.run_sync(sample_snapshots, "Yes Bank", magnitude=0.75)
    assert isinstance(result, MCResult)


def test_mc_run_count(sample_snapshots):
    """MCResult reports correct total_runs."""
    mc     = MonteCarloStressTester(n_runs=50)
    result = mc.run_sync(sample_snapshots, "Yes Bank", magnitude=0.75)
    assert result.total_runs == 50


def test_sri_distribution_shape(sample_snapshots):
    """SRI distribution has correct length."""
    mc     = MonteCarloStressTester(n_runs=50)
    result = mc.run_sync(sample_snapshots, "Yes Bank", magnitude=0.75)
    assert len(result.sri_distribution) == 50


def test_sri_values_in_range(sample_snapshots):
    """All SRI values in [0, 1]."""
    mc     = MonteCarloStressTester(n_runs=50)
    result = mc.run_sync(sample_snapshots, "Yes Bank", magnitude=0.75)
    assert np.all(result.sri_distribution >= 0.0)
    assert np.all(result.sri_distribution <= 1.0)


def test_ci_valid(sample_snapshots):
    """CI low <= mean <= CI high."""
    mc     = MonteCarloStressTester(n_runs=50)
    result = mc.run_sync(sample_snapshots, "Yes Bank", magnitude=0.75)
    assert result.sri_ci_low  <= result.sri_mean
    assert result.sri_mean    <= result.sri_ci_high


def test_worst_case_gte_mean(sample_snapshots):
    """Worst-case SRI >= mean SRI."""
    mc     = MonteCarloStressTester(n_runs=50)
    result = mc.run_sync(sample_snapshots, "Yes Bank", magnitude=0.75)
    assert result.worst_case_sri >= result.sri_mean


def test_p_dsib_in_range(sample_snapshots):
    """P(D-SIB affected) is in [0, 1]."""
    mc     = MonteCarloStressTester(n_runs=50)
    result = mc.run_sync(sample_snapshots, "Yes Bank", magnitude=0.75)
    assert 0.0 <= result.p_dsib_affected <= 1.0


def test_recovery_days_positive_when_high_sri(sample_snapshots):
    """Recovery days > 0 when SRI is high."""
    mc     = MonteCarloStressTester(n_runs=50)
    result = mc.run_sync(sample_snapshots, "Yes Bank", magnitude=0.90)
    if result.sri_mean > 0.5:
        assert result.recovery_rate > 0


def test_mc_async_nonblocking(sample_snapshots):
    """run_async returns immediately (non-blocking)."""
    import time
    mc     = MonteCarloStressTester(n_runs=100)
    start  = time.time()
    mc.run_async(sample_snapshots, "Yes Bank", magnitude=0.75)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"run_async should return in < 1s, took {elapsed:.2f}s"


def test_get_last_result_none_before_run():
    """get_last_result returns None before any run."""
    mc = MonteCarloStressTester(n_runs=10)
    assert mc.get_last_result() is None
