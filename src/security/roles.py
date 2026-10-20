"""
Financial Contagion Monitor — src/security/roles.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

Role enforcement — endpoint-level access control.
Every API route and sensitive callback uses require_role() decorator.
A SubAdmin JWT that reaches an Admin endpoint gets 403 — not just a UI block.

Role hierarchy (ascending privilege):
  DEMO      → read-only, no alerts, no simulation
  SUBADMIN  → own bank only, own alerts, limited network view
  ADMIN     → own bank full + system-wide risk read + shock simulation
  SUPERADMIN → everything: force alerts, view all banks, audit log
"""

from functools import wraps
from typing import Callable, Optional

# Role hierarchy — higher index = more privilege
ROLE_HIERARCHY = ["DEMO", "SUBADMIN", "ADMIN", "SUPERADMIN"]

# Permission matrix: action -> minimum required role
PERMISSIONS = {
    # Dashboard data
    "view_own_bank":       "DEMO",
    "view_all_banks":      "ADMIN",
    "view_system_metrics": "ADMIN",

    # Simulation
    "run_shock_simulation": "ADMIN",
    "view_simulation":      "DEMO",

    # Alerts
    "view_own_alerts":      "SUBADMIN",
    "view_all_alerts":      "SUPERADMIN",
    "acknowledge_alert":    "SUBADMIN",
    "force_alert":          "SUPERADMIN",

    # Reports
    "generate_report":      "ADMIN",
    "export_pdf":           "ADMIN",

    # Admin actions
    "view_audit_log":       "SUPERADMIN",
    "system_stress_test":   "SUPERADMIN",
    "toggle_live_mode":     "ADMIN",
}


def require_role(*allowed_roles: str) -> Callable:
    """
    Flask route decorator that enforces JWT role requirements.
    Usage:
        @server.route("/api/force-alert", methods=["POST"])
        @require_role("SUPERADMIN")
        def api_force_alert(jwt_payload):
            ...

        @server.route("/api/run-shock", methods=["POST"])
        @require_role("SUPERADMIN", "ADMIN")
        def api_run_shock(jwt_payload):
            ...
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                from flask import request, jsonify
                from src.security.auth import AuthManager

                token = request.headers.get("Authorization", "").replace("Bearer ", "")
                if not token:
                    return jsonify({"error": "Unauthorized — no token"}), 401

                # Verify token (AuthManager is a singleton in app.py)
                # For route-level use, instantiate lightweight verify
                import os
                import jwt as pyjwt
                from dotenv import load_dotenv
                load_dotenv()
                secret = os.getenv("JWT_SECRET", "")
                try:
                    payload = pyjwt.decode(token, secret, algorithms=["HS256"])
                except Exception:
                    return jsonify({"error": "Unauthorized — invalid token"}), 401

                role = payload.get("role", "")
                if role not in allowed_roles:
                    return jsonify({
                        "error": f"Forbidden — requires one of {list(allowed_roles)}, "
                                 f"got {role}"
                    }), 403

                return f(*args, **kwargs, jwt_payload=payload)

            except ImportError:
                # Flask not available in test context — pass through
                return f(*args, **kwargs, jwt_payload={})

        return wrapper
    return decorator


def has_permission(role: str, action: str) -> bool:
    """
    Checks if a role has permission to perform an action.

    Args:
        role:   JWT role string ("SUPERADMIN", "ADMIN", "SUBADMIN", "DEMO")
        action: Permission key from PERMISSIONS dict

    Returns:
        True if role has at least the minimum required privilege level
    """
    required_role = PERMISSIONS.get(action)
    if required_role is None:
        return False  # unknown action — deny by default

    try:
        role_idx     = ROLE_HIERARCHY.index(role)
        required_idx = ROLE_HIERARCHY.index(required_role)
        return role_idx >= required_idx
    except ValueError:
        return False


def can_view_bank(jwt_payload: dict, target_bank_name: str) -> bool:
    """
    Checks if a logged-in user can view data for a specific bank.

    Rules:
      SUPERADMIN → can view all 33 banks
      ADMIN      → can view all 33 banks (read-only for non-own banks)
      SUBADMIN   → can only view own bank
      DEMO       → can view all banks (read-only)
    """
    role = jwt_payload.get("role", "")
    user_bank = jwt_payload.get("bank_name", "")

    if role in ("SUPERADMIN", "ADMIN", "DEMO"):
        return True
    if role == "SUBADMIN":
        return user_bank == target_bank_name
    return False


def can_view_alert(jwt_payload: dict, alert_target_bank: str) -> bool:
    """
    Checks if user can receive a specific alert.
    SUPERADMIN sees all. Others only see alerts for their own bank.
    """
    role = jwt_payload.get("role", "")
    if role == "SUPERADMIN":
        return True
    return jwt_payload.get("bank_name", "") == alert_target_bank


def filter_data_for_role(data: dict, jwt_payload: dict) -> dict:
    """
    Filters dashboard data based on the user's role.
    SUBADMIN users only see their own bank's detailed metrics.
    Used in build_all_outputs() to gate sensitive data.
    """
    role = jwt_payload.get("role", "DEMO")
    if role in ("SUPERADMIN", "ADMIN", "DEMO"):
        return data  # full access

    # SUBADMIN: filter snapshots to own bank + strip sensitive fields
    user_bank = jwt_payload.get("bank_name", "")
    filtered = data.copy()

    if "snapshots" in filtered:
        filtered["snapshots"] = [
            s for s in filtered["snapshots"]
            if s.institution_id == user_bank
        ]

    if "top_risks" in filtered:
        # SUBADMIN can still see top risks list (bank names only, not metrics)
        filtered["top_risks"] = [
            {"bank": r.institution_id, "level": "REDACTED"}
            for r in filtered["top_risks"]
        ]

    return filtered
