"""tests/test_security.py — 10 tests for AuthManager, JWT, rate limiting, lockout."""
import pytest
import time

from src.security.auth import AuthManager


@pytest.fixture
def auth(tmp_db):
    a = AuthManager(tmp_db)
    a.initialize_all_users()
    return a


def test_initialize_creates_35_users(auth, tmp_db):
    """initialize_all_users creates exactly 35 users."""
    import sqlite3
    conn  = sqlite3.connect(tmp_db)
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    assert count == 35


def test_login_success(auth):
    """Valid credentials return success=True with token."""
    result = auth.login("rbi_superadmin", "RBI@SuperAdmin#2026!", "127.0.0.1")
    assert result["success"] is True
    assert "token" in result
    assert result["role"] == "SUPERADMIN"


def test_login_wrong_password(auth):
    """Wrong password returns success=False."""
    result = auth.login("sbi_admin", "WRONG_PASSWORD", "127.0.0.1")
    assert result["success"] is False
    assert "error" in result


def test_login_unknown_user(auth):
    """Unknown username returns success=False."""
    result = auth.login("nobody_user", "anypassword", "127.0.0.1")
    assert result["success"] is False


def test_jwt_verify_valid(auth):
    """verify_token returns payload for valid JWT."""
    result  = auth.login("sbi_admin", "SBI@Admin#FCM2026!", "127.0.0.1")
    token   = result["token"]
    payload = auth.verify_token(token)
    assert payload is not None
    assert payload["role"] == "ADMIN"
    assert payload["bank_name"] == "State Bank of India (SBI)"


def test_jwt_verify_invalid(auth):
    """verify_token returns None for garbage token."""
    assert auth.verify_token("not.a.valid.token") is None


def test_jwt_verify_empty(auth):
    """verify_token returns None for empty string."""
    assert auth.verify_token("") is None


def test_account_lockout_after_5_failures(auth):
    """Account is locked after 5 failed login attempts."""
    for _ in range(5):
        auth.login("yes_user", "WRONG_PW", "192.168.1.1")
    result = auth.login("yes_user", "WRONG_PW", "192.168.1.1")
    # Should be locked now
    assert result["success"] is False
    assert "lock" in result["error"].lower() or "attempt" in result["error"].lower()


def test_initialize_idempotent(auth, tmp_db):
    """Calling initialize_all_users twice does not duplicate users."""
    import sqlite3
    auth.initialize_all_users()  # call again
    conn  = sqlite3.connect(tmp_db)
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    assert count == 35


def test_audit_log_created(auth):
    """Audit log file is created after login."""
    import os
    auth.login("demo_user", "Demo@ReadOnly#2026!", "127.0.0.1")
    assert os.path.exists("data/security/audit.log")
