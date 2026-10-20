"""
Financial Contagion Monitor — src/alerts/manager.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

AlertManager — shared real-time alert bus backed by SQLite.

The multi-screen alert cascade demo (Feature 6) depends entirely on this module.
When three laptops are on the same WiFi and all connected to the Dash server:
  1. RBI fires force_alert → Yes Bank
  2. Yes Bank screen goes red within 3 seconds (polling interval)
  3. RBL + IndusInd screens go red if cascade threshold crossed
  4. SBI screen stays normal (alert was not for SBI)
  5. RBI sees all alerts in priority queue

Alert severity thresholds:
  LOW       → vulnerability > 0.50 OR npl > 0.12
  ELEVATED  → vulnerability > 0.65 OR cds > 300bps
  CRITICAL  → vulnerability > 0.80 OR SRI > 0.70
  COLLAPSE  → vulnerability > 0.92 OR distress_event == 1

Alerts ONLY fire in Live Market Mode.
In Historical Replay Mode, check_and_fire() is a no-op.
"""

import sqlite3
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from src.data.loader import DayResult, BankSnapshot
from src.ews.detector import EarlyWarningDetector

# ── Alert Schema ──────────────────────────────────────────────────────────────
CREATE_ALERTS_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    target_bank     TEXT NOT NULL,
    severity        TEXT NOT NULL,
    title           TEXT NOT NULL,
    message         TEXT NOT NULL,
    source          TEXT NOT NULL,
    triggered_at    TEXT NOT NULL,
    acknowledged    INTEGER DEFAULT 0,
    acknowledged_by TEXT,
    acknowledged_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_bank ON alerts(target_bank);
CREATE INDEX IF NOT EXISTS idx_alerts_acked ON alerts(acknowledged);
"""

# Severity thresholds
SEVERITY_THRESHOLDS = {
    "LOW":      {"vuln": 0.50, "npl": 0.12, "cds": None,  "sri": None},
    "ELEVATED": {"vuln": 0.65, "npl": None,  "cds": 300.0, "sri": None},
    "CRITICAL": {"vuln": 0.80, "npl": None,  "cds": None,  "sri": 0.70},
    "COLLAPSE": {"vuln": 0.92, "npl": None,  "cds": None,  "sri": None},
}


@dataclass
class Alert:
    alert_id:        int
    target_bank:     str
    severity:        str
    title:           str
    message:         str
    source:          str
    triggered_at:    str
    acknowledged:    int
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[str]


class AlertManager:
    """
    Manages the shared alert bus used by all connected browser sessions.

    Usage:
        manager = AlertManager("data/security/users.db", ews)
        # In live update callback:
        manager.check_and_fire(day_result, mode="live")
        # In alert polling endpoint:
        alerts = manager.get_alerts("Yes Bank", since="2026-01-01T00:00:00")
    """

    def __init__(
        self,
        db_path: str = "data/security/users.db",
        ews: Optional[EarlyWarningDetector] = None,
    ):
        self.db_path = db_path
        self._ews = ews
        self._init_alerts_table()

        # Deduplication: track recently fired alerts to avoid spamming
        # {(bank_name, severity): last_fired_datetime}
        self._recently_fired: dict = {}
        self._dedup_window_seconds = 300  # 5 minutes

    # ── Public: Threshold Checking ────────────────────────────────────────────

    def check_and_fire(self, day_result: DayResult, mode: str = "historical") -> int:
        """
        Checks all 33 bank snapshots against severity thresholds.
        Creates alert records for any that breach thresholds.

        CRITICAL: mode must be "live" to fire alerts.
        Historical replay mode NEVER fires alerts.

        Args:
            day_result: Current DayResult (from LiveDataLoader or ReplayEngine)
            mode:       "live" or "historical"

        Returns:
            Number of new alerts fired (0 in historical mode)
        """
        if mode != "live":
            return 0  # Alerts disabled in historical replay — by design

        fired = 0
        for snap in day_result.snapshots:
            severity = self._assess_severity(snap, day_result.sri)
            if severity is None:
                continue

            # Deduplication check
            dedup_key = (snap.institution_id, severity)
            last_fired = self._recently_fired.get(dedup_key)
            if last_fired:
                elapsed = (datetime.now() - last_fired).total_seconds()
                if elapsed < self._dedup_window_seconds:
                    continue  # already fired recently

            title, message = self._build_alert_message(snap, severity, day_result)
            self._insert_alert(
                target_bank=snap.institution_id,
                severity=severity,
                title=title,
                message=message,
                source="LIVE_DATA",
            )
            self._recently_fired[dedup_key] = datetime.now()
            fired += 1

        return fired

    def get_alerts(
        self,
        bank_name: str,
        since: str = "2000-01-01T00:00:00",
    ) -> List[Alert]:
        """
        Returns unacknowledged alerts for a specific bank since a timestamp.
        Called by the 3-second polling endpoint for each browser session.

        Args:
            bank_name: Full institution_id (from JWT payload)
            since:     ISO timestamp — only return alerts newer than this
        """
        conn = self._connect()
        rows = conn.execute("""
            SELECT alert_id, target_bank, severity, title, message,
                   source, triggered_at, acknowledged,
                   acknowledged_by, acknowledged_at
            FROM alerts
            WHERE target_bank = ?
              AND acknowledged = 0
              AND triggered_at > ?
            ORDER BY triggered_at DESC
            LIMIT 50
        """, (bank_name, since)).fetchall()
        conn.close()
        return [Alert(*row) for row in rows]

    def get_all_alerts(self, limit: int = 200) -> List[Alert]:
        """
        Returns all unacknowledged alerts across all banks.
        SUPERADMIN only — enforced at the API route level via require_role().
        """
        conn = self._connect()
        rows = conn.execute("""
            SELECT alert_id, target_bank, severity, title, message,
                   source, triggered_at, acknowledged,
                   acknowledged_by, acknowledged_at
            FROM alerts
            WHERE acknowledged = 0
            ORDER BY
                CASE severity
                    WHEN 'COLLAPSE'  THEN 1
                    WHEN 'CRITICAL'  THEN 2
                    WHEN 'ELEVATED'  THEN 3
                    WHEN 'LOW'       THEN 4
                    ELSE 5
                END,
                triggered_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()
        return [Alert(*row) for row in rows]

    def acknowledge(self, alert_id: int, bank_name: str) -> bool:
        """
        Marks an alert as acknowledged.
        Verifies that the alert's target_bank matches bank_name (security check).
        SUPERADMIN can acknowledge any alert.

        Returns True if acknowledged, False if not found or not authorized.
        """
        conn = self._connect()
        # Check ownership
        row = conn.execute(
            "SELECT target_bank FROM alerts WHERE alert_id = ?", (alert_id,)
        ).fetchone()

        if row is None:
            conn.close()
            return False

        target = row[0]
        # Allow if it's your bank OR if you're the RBI (bank_name="Reserve Bank of India")
        if target != bank_name and bank_name != "Reserve Bank of India":
            conn.close()
            return False

        conn.execute("""
            UPDATE alerts
            SET acknowledged = 1,
                acknowledged_by = ?,
                acknowledged_at = ?
            WHERE alert_id = ?
        """, (bank_name, datetime.now().isoformat(), alert_id))
        conn.commit()
        conn.close()
        return True

    def force_alert(
        self,
        target_bank: str,
        message: str,
        severity: str = "CRITICAL",
        source: str = "RBI_MANUAL",
    ) -> int:
        """
        SUPERADMIN-only: manually injects an alert for any bank.
        Used in the multi-screen demo and for the panel demo sequence.

        Returns:
            alert_id of the newly created alert
        """
        severity = severity.upper()
        if severity not in ("LOW", "ELEVATED", "CRITICAL", "COLLAPSE"):
            severity = "CRITICAL"

        title = f"RBI MANUAL ALERT — {severity}"
        alert_id = self._insert_alert(
            target_bank=target_bank,
            severity=severity,
            title=title,
            message=message,
            source=source,
        )
        return alert_id

    def fire_simulation_alert(
        self,
        target_bank: str,
        cascade_step: int,
        shock_bank: str,
        magnitude: float,
    ) -> Optional[int]:
        """
        Fires a simulation warning alert.
        Only fires if magnitude >= 0.60 AND data mode is live (checked by caller).
        Alert message includes "SIMULATION WARNING" to distinguish from real alerts.
        """
        title = f"SIMULATION WARNING — Cascade reached {target_bank}"
        message = (
            f"SIMULATION WARNING: Shock originating at {shock_bank} "
            f"(magnitude {magnitude:.0%}) has propagated to {target_bank} "
            f"at cascade step T+{cascade_step}. This is a simulation — not live data."
        )
        return self._insert_alert(
            target_bank=target_bank,
            severity="ELEVATED",
            title=title,
            message=message,
            source="SIMULATION",
        )

    # ── Private: Severity Assessment ──────────────────────────────────────────

    def _assess_severity(
        self,
        snap: BankSnapshot,
        system_sri: float,
    ) -> Optional[str]:
        """
        Determines if a bank snapshot breaches any alert threshold.
        Returns the highest applicable severity, or None if below all thresholds.
        """
        vuln = snap.vulnerability_label_injected
        npl  = snap.npl_ratio
        cds  = snap.cds_spread_bps
        distress = snap.distress_event_injected

        # Check from most severe to least
        if vuln >= 0.92 or distress == 1:
            return "COLLAPSE"
        if vuln >= 0.80 or system_sri >= 0.70:
            return "CRITICAL"
        if vuln >= 0.65 or cds >= 300.0:
            return "ELEVATED"
        if vuln >= 0.50 or npl >= 0.12:
            return "LOW"
        return None

    def _build_alert_message(
        self,
        snap: BankSnapshot,
        severity: str,
        day_result: DayResult,
    ):
        """Builds title and message strings for an alert."""
        bank = snap.bank_short or snap.institution_id

        title_map = {
            "COLLAPSE":  f"🚨 COLLAPSE WARNING — {bank}",
            "CRITICAL":  f"⚠ CRITICAL RISK — {bank}",
            "ELEVATED":  f"▲ ELEVATED RISK — {bank}",
            "LOW":       f"ℹ RISK MONITOR — {bank}",
        }
        title = title_map.get(severity, f"ALERT — {bank}")

        message = (
            f"{snap.institution_id} has breached {severity} threshold. "
            f"Vulnerability: {snap.vulnerability_label_injected:.3f} | "
            f"NPL: {snap.npl_ratio:.3f} | "
            f"CDS: {snap.cds_spread_bps:.0f}bps | "
            f"System SRI: {day_result.sri:.3f} | "
            f"Date: {day_result.date}"
        )

        if day_result.crisis_active:
            message += f" | Active Crisis: {day_result.crisis_name}"

        return title, message

    # ── Private: DB ──────────────────────────────────────────────────────────

    def _init_alerts_table(self) -> None:
        """Creates the alerts table if it doesn't exist."""
        conn = self._connect()
        conn.executescript(CREATE_ALERTS_TABLE)
        conn.commit()
        conn.close()

    def _insert_alert(
        self,
        target_bank: str,
        severity: str,
        title: str,
        message: str,
        source: str,
    ) -> int:
        """Inserts a new alert row and returns its alert_id."""
        conn = self._connect()
        cursor = conn.execute("""
            INSERT INTO alerts
                (target_bank, severity, title, message, source, triggered_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            target_bank, severity, title, message, source,
            datetime.now().isoformat()
        ))
        conn.commit()
        alert_id = cursor.lastrowid
        conn.close()
        return alert_id

    def _connect(self) -> sqlite3.Connection:
        """Opens SQLite connection with WAL mode for concurrent multi-session access."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
