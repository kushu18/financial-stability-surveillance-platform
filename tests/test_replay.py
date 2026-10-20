"""tests/test_replay.py — 10 tests for ReplayEngine seek/step/crisis navigation."""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from src.replay.engine import ReplayEngine
from src.data.loader import RealDataLoader


def _make_mock_engine(sample_df_one_date):
    """Creates a ReplayEngine with mocked loader (no file I/O)."""
    loader = MagicMock(spec=RealDataLoader)
    loader.load_generated.return_value = sample_df_one_date
    loader.get_sri_series.return_value = pd.DataFrame({
        "date": ["2020-03-15"],
        "sri":  [0.92],
    })
    engine = ReplayEngine(loader)
    engine.load()
    return engine


def test_engine_loads(sample_df_one_date):
    """ReplayEngine.load() completes without error."""
    engine = _make_mock_engine(sample_df_one_date)
    assert engine._loaded is True


def test_seek_returns_day_result(sample_df_one_date):
    """seek() returns a DayResult for a valid date."""
    from src.data.loader import DayResult
    engine = _make_mock_engine(sample_df_one_date)
    result = engine.seek("2020-03-15")
    assert isinstance(result, DayResult)
    assert result.date == "2020-03-15"


def test_seek_covid_sri_high(sample_df_one_date):
    """Seeking 2020-03-15 returns SRI > 0.70 (asset-weighted)."""
    engine = _make_mock_engine(sample_df_one_date)
    result = engine.seek("2020-03-15")
    assert result.sri > 0.70, f"COVID SRI should be > 0.70, got {result.sri:.4f}"


def test_seek_covid_crisis_active(sample_df_one_date):
    """COVID crisis is active on 2020-03-15."""
    engine = _make_mock_engine(sample_df_one_date)
    result = engine.seek("2020-03-15")
    assert result.crisis_active is True
    assert "COVID" in result.crisis_name


def test_seek_snapshots_count(sample_df_one_date):
    """seek() returns 33 bank snapshots."""
    engine = _make_mock_engine(sample_df_one_date)
    result = engine.seek("2020-03-15")
    assert len(result.snapshots) == 33


def test_seek_top_risks_not_empty(sample_df_one_date):
    """top_risks is populated after seek."""
    engine = _make_mock_engine(sample_df_one_date)
    result = engine.seek("2020-03-15")
    assert len(result.top_risks) > 0


def test_weekend_snap(sample_df_one_date):
    """Seeking a weekend date snaps to nearest available date."""
    engine = _make_mock_engine(sample_df_one_date)
    # 2020-03-14 is Saturday
    result = engine.seek("2020-03-14")
    # Should snap to an available date
    assert result.date in engine.get_available_dates()


def test_available_dates_sorted(sample_df_one_date):
    """get_available_dates() returns sorted list."""
    engine = _make_mock_engine(sample_df_one_date)
    dates  = engine.get_available_dates()
    assert dates == sorted(dates)


def test_step_forward(sample_df_one_date):
    """step(1) returns valid DayResult."""
    from src.data.loader import DayResult
    engine = _make_mock_engine(sample_df_one_date)
    engine.seek("2020-03-15")
    result = engine.step(1)
    assert isinstance(result, DayResult)


def test_jump_to_crisis(sample_df_one_date):
    """jump_to_crisis raises ValueError for unknown crisis name."""
    engine = _make_mock_engine(sample_df_one_date)
    with pytest.raises(ValueError):
        engine.jump_to_crisis("NONEXISTENT CRISIS 9999")
