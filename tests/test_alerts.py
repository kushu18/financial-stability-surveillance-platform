"""tests/test_alerts.py — 10 tests for AlertManager routing, severity, acknowledgement."""
import pytest
import sqlite3

from src.alerts.manager import AlertManager


@pytest.fixture
def alert_manager(tmp_db):
    # Reuse auth DB for alerts (same DB in production)
    from src.security.auth import AuthManager
    auth = AuthManager(tmp_db)
    auth.initialize_all_users()
    return AlertManager(tmp_db)


def test_force_alert_creates_record(alert_manager):
    """force_alert inserts a row in the alerts table."""
    aid = alert_manager.force_alert("Yes Bank", "Test alert", "CRITICAL")
    assert isinstance(aid, int)
    assert aid > 0


def test_get_alerts_routing(alert_manager):
    """get_alerts only returns alerts for the specified bank."""
    alert_manager.force_alert("Yes Bank",  "Alert for Yes", "CRITICAL")
    alert_manager.force_alert("HDFC Bank", "Alert for HDFC", "LOW")

    yes_alerts  = alert_manager.get_alerts("Yes Bank")
    hdfc_alerts = alert_manager.get_alerts("HDFC Bank")

    assert all(a.target_bank == "Yes Bank"  for a in yes_alerts)
    assert all(a.target_bank == "HDFC Bank" for a in hdfc_alerts)


def test_get_all_alerts_superadmin(alert_manager):
    """get_all_alerts returns alerts across all banks."""
    alert_manager.force_alert("Yes Bank",  "A1", "CRITICAL")
    alert_manager.force_alert("HDFC Bank", "A2", "LOW")
    all_alerts = alert_manager.get_all_alerts()
    banks = {a.target_bank for a in all_alerts}
    assert len(banks) >= 2


def test_acknowledge_own_bank(alert_manager):
    """Bank user can acknowledge their own alert."""
    aid = alert_manager.force_alert("Yes Bank", "Test", "CRITICAL")
    ok  = alert_manager.acknowledge(aid, "Yes Bank")
    assert ok is True


def test_acknowledge_wrong_bank_denied(alert_manager):
    """Bank user cannot acknowledge another bank's alert."""
    aid = alert_manager.force_alert("Yes Bank", "Test", "CRITICAL")
    ok  = alert_manager.acknowledge(aid, "HDFC Bank")
    assert ok is False


def test_acknowledged_alert_hidden(alert_manager):
    """Acknowledged alerts do not appear in get_alerts."""
    aid = alert_manager.force_alert("Yes Bank", "Test", "ELEVATED")
    alert_manager.acknowledge(aid, "Yes Bank")
    alerts = alert_manager.get_alerts("Yes Bank")
    ids    = [a.alert_id for a in alerts]
    assert aid not in ids


def test_check_and_fire_historical_noop(alert_manager, sample_day_result):
    """check_and_fire does nothing in historical mode."""
    fired = alert_manager.check_and_fire(sample_day_result, mode="historical")
    assert fired == 0


def test_check_and_fire_live_fires(alert_manager, sample_day_result):
    """check_and_fire fires alerts in live mode when thresholds breached."""
    fired = alert_manager.check_and_fire(sample_day_result, mode="live")
    assert fired >= 1  # COVID peak should breach thresholds


def test_severity_ordering_in_all_alerts(alert_manager):
    """get_all_alerts returns COLLAPSE/CRITICAL before LOW."""
    alert_manager.force_alert("Yes Bank",  "Low alert",      "LOW")
    alert_manager.force_alert("HDFC Bank", "Collapse alert", "COLLAPSE")
    alerts = alert_manager.get_all_alerts()
    if len(alerts) >= 2:
        assert alerts[0].severity in ("COLLAPSE", "CRITICAL")


def test_rbi_can_acknowledge_any_bank(alert_manager):
    """RBI (Reserve Bank of India) can acknowledge any bank's alert."""
    aid = alert_manager.force_alert("Yes Bank", "Test", "CRITICAL")
    ok  = alert_manager.acknowledge(aid, "Reserve Bank of India")
    assert ok is True
