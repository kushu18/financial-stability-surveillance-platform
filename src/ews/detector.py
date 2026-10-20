"""
Financial Contagion Monitor — src/ews/detector.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

EarlyWarningDetector — multi-threshold early warning system that monitors
the rolling 7-day SRI trend and individual bank metrics.

Demo requirement:
  When seeking to 2020-02-14 (one month before COVID peak),
  EWS must already show ELEVATED — not NORMAL.
  This is because Yes Bank Stress crisis was active from 2019-03-05
  and several banks were showing elevated NPL and CDS signals.

Warning levels (ascending severity):
  NORMAL    → baseline, no action
  ELEVATED  → monitoring required, amber banner
  CRITICAL  → immediate attention, red banner + alerts fire

Used by:
  ReplayEngine.seek()         — sets DayResult.ews_level
  LiveDataLoader.get_current() — same
  AlertManager.check_and_fire() — uses level to decide alert severity
  ReportGenerator             — section 1 of LLM report
"""

from collections import deque
from typing import List, Optional

from src.data.loader import DayResult, BankSnapshot


class EarlyWarningDetector:
    """
    Stateful detector — maintains a rolling window of recent DayResults
    to compute trend direction and level transitions.

    Usage:
        ews = EarlyWarningDetector()
        level = ews.update(day_result)   # returns "NORMAL"/"ELEVATED"/"CRITICAL"
        trend = ews.get_trend()           # "RISING"/"FALLING"/"STABLE"
        triggers = ews.get_triggered_thresholds()  # ["SRI > 0.65", "NPL > 0.18"]
    """

    # Threshold definitions per level
    # A level is triggered if ANY of its conditions are met
    THRESHOLDS = {
        "NORMAL": {
            "sri":  0.00,
            "npl":  0.08,
            "cds":  150.0,
            "vuln": 0.30,
        },
        "ELEVATED": {
            "sri":  0.45,
            "npl":  0.12,
            "cds":  250.0,
            "vuln": 0.55,
        },
        "CRITICAL": {
            "sri":  0.65,
            "npl":  0.18,
            "cds":  400.0,
            "vuln": 0.80,
        },
    }

    # Hysteresis: level won't drop unless value is below threshold - buffer
    # Prevents rapid oscillation around a threshold boundary
    HYSTERESIS_BUFFER = {
        "sri":  0.05,
        "npl":  0.02,
        "cds":  30.0,
        "vuln": 0.05,
    }

    TREND_WINDOW = 7   # days for trend computation
    MAX_HISTORY  = 30  # max days of history to retain

    def __init__(self):
        self._current_level: str = "NORMAL"
        self._sri_history: deque = deque(maxlen=self.MAX_HISTORY)
        self._triggered_thresholds: List[str] = []
        self._consecutive_elevated: int = 0   # for persistence logic
        self._last_date: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, day_result: DayResult) -> str:
        """
        Processes a new DayResult and returns the current warning level.

        Args:
            day_result: Latest DayResult from ReplayEngine or LiveDataLoader

        Returns:
            Warning level string: "NORMAL", "ELEVATED", or "CRITICAL"
        """
        # Update SRI history
        self._sri_history.append({
            "date": day_result.date,
            "sri":  day_result.sri,
        })
        self._last_date = day_result.date

        # Compute aggregate metrics from snapshots
        metrics = self._compute_metrics(day_result)

        # Determine triggered thresholds and new level
        new_level, triggered = self._evaluate_level(metrics)

        # Apply persistence: don't drop level if we've been elevated for < 3 days
        # This prevents a single day of good data from clearing an alert
        if new_level == "NORMAL" and self._current_level != "NORMAL":
            self._consecutive_elevated += 1
            if self._consecutive_elevated < 3:
                new_level = self._current_level  # hold current level
        else:
            self._consecutive_elevated = 0

        self._current_level = new_level
        self._triggered_thresholds = triggered

        return self._current_level

    def get_triggered_thresholds(self) -> List[str]:
        """
        Returns human-readable list of currently triggered threshold conditions.
        Example: ["SRI > 0.65", "Max NPL > 0.18", "2 banks CDS > 400bps"]
        """
        return self._triggered_thresholds.copy()

    def get_trend(self, window: int = 7) -> str:
        """
        Returns SRI trend direction over the last `window` days.

        Returns:
            "RISING"  — SRI increased by > 0.02 over window
            "FALLING" — SRI decreased by > 0.02 over window
            "STABLE"  — change within ±0.02
        """
        if len(self._sri_history) < 2:
            return "STABLE"

        actual_window = min(window, len(self._sri_history))
        history_list = list(self._sri_history)

        recent_sri = history_list[-1]["sri"]
        window_start_sri = history_list[-actual_window]["sri"]
        delta = recent_sri - window_start_sri

        if delta > 0.02:
            return "RISING"
        elif delta < -0.02:
            return "FALLING"
        else:
            return "STABLE"

    def get_current_level(self) -> str:
        """Returns the current warning level."""
        return self._current_level

    def get_sri_trend_values(self) -> List[float]:
        """Returns list of recent SRI values for mini-sparkline in dashboard."""
        return [h["sri"] for h in self._sri_history]

    def reset(self) -> None:
        """Resets detector state. Called when switching between modes."""
        self._current_level = "NORMAL"
        self._sri_history.clear()
        self._triggered_thresholds = []
        self._consecutive_elevated = 0
        self._last_date = ""

    # ── Private: Metric Computation ───────────────────────────────────────────

    def _compute_metrics(self, day_result: DayResult) -> dict:
        """
        Extracts key metrics from a DayResult for threshold evaluation.

        Returns dict with:
          sri       — current Systemic Risk Index
          max_npl   — maximum NPL ratio across all 33 banks
          max_cds   — maximum CDS spread across all 33 banks
          max_vuln  — maximum vulnerability score across all 33 banks
          n_distress — number of banks with distress_event_injected == 1
          n_elevated — number of banks with vulnerability > ELEVATED.vuln threshold
        """
        snapshots = day_result.snapshots
        if not snapshots:
            return {
                "sri": day_result.sri,
                "max_npl": 0.0,
                "max_cds": 0.0,
                "max_vuln": 0.0,
                "n_distress": 0,
                "n_elevated": 0,
            }

        npl_values  = [s.npl_ratio for s in snapshots]
        cds_values  = [s.cds_spread_bps for s in snapshots]
        vuln_values = [s.vulnerability_label_injected for s in snapshots]

        elevated_vuln_threshold = self.THRESHOLDS["ELEVATED"]["vuln"]

        return {
            "sri":        day_result.sri,
            "max_npl":    max(npl_values),
            "max_cds":    max(cds_values),
            "max_vuln":   max(vuln_values),
            "n_distress": sum(1 for s in snapshots if s.distress_event_injected == 1),
            "n_elevated": sum(1 for v in vuln_values if v > elevated_vuln_threshold),
        }

    def _evaluate_level(self, metrics: dict):
        """
        Evaluates metrics against THRESHOLDS and returns (level, triggered_list).

        Levels are checked from most severe to least.
        Returns the highest level whose conditions are met.
        """
        triggered = []
        determined_level = "NORMAL"

        # Check CRITICAL first
        crit = self.THRESHOLDS["CRITICAL"]
        if metrics["sri"] >= crit["sri"]:
            triggered.append(f"SRI {metrics['sri']:.3f} ≥ {crit['sri']}")
            determined_level = "CRITICAL"
        if metrics["max_npl"] >= crit["npl"]:
            triggered.append(f"Max NPL {metrics['max_npl']:.3f} ≥ {crit['npl']}")
            determined_level = "CRITICAL"
        if metrics["max_cds"] >= crit["cds"]:
            triggered.append(f"Max CDS {metrics['max_cds']:.0f}bps ≥ {crit['cds']:.0f}")
            determined_level = "CRITICAL"
        if metrics["max_vuln"] >= crit["vuln"]:
            triggered.append(f"Max vulnerability {metrics['max_vuln']:.3f} ≥ {crit['vuln']}")
            determined_level = "CRITICAL"

        if determined_level == "CRITICAL":
            return "CRITICAL", triggered

        # Check ELEVATED
        elev = self.THRESHOLDS["ELEVATED"]
        if metrics["sri"] >= elev["sri"]:
            triggered.append(f"SRI {metrics['sri']:.3f} ≥ {elev['sri']}")
            determined_level = "ELEVATED"
        if metrics["max_npl"] >= elev["npl"]:
            triggered.append(f"Max NPL {metrics['max_npl']:.3f} ≥ {elev['npl']}")
            determined_level = "ELEVATED"
        if metrics["max_cds"] >= elev["cds"]:
            triggered.append(f"Max CDS {metrics['max_cds']:.0f}bps ≥ {elev['cds']:.0f}")
            determined_level = "ELEVATED"
        if metrics["max_vuln"] >= elev["vuln"]:
            triggered.append(f"Max vulnerability {metrics['max_vuln']:.3f} ≥ {elev['vuln']}")
            determined_level = "ELEVATED"
        if metrics["n_distress"] >= 2:
            triggered.append(f"{metrics['n_distress']} banks in active distress")
            determined_level = "ELEVATED"

        # Active crisis in dataset also elevates
        # (handled at DayResult level by get_active_crisis in loader.py)

        if determined_level == "ELEVATED":
            return "ELEVATED", triggered

        return "NORMAL", []

    def get_ews_banner_text(self) -> str:
        """Returns formatted banner text for the dashboard EWS component."""
        level = self._current_level
        trend = self.get_trend()
        triggers = self._triggered_thresholds

        if level == "NORMAL":
            return "EWS: NORMAL — All indicators within safe thresholds"

        trigger_str = " | ".join(triggers[:2]) if triggers else ""
        trend_str = f"Trend: {trend}"

        if level == "CRITICAL":
            return f"⚠ EWS: CRITICAL — {trigger_str} | {trend_str}"
        else:
            return f"▲ EWS: ELEVATED — {trigger_str} | {trend_str}"

    def get_alert_severity_from_level(self) -> str:
        """Maps EWS level to AlertManager severity string."""
        mapping = {
            "NORMAL":   "LOW",
            "ELEVATED": "ELEVATED",
            "CRITICAL": "CRITICAL",
        }
        return mapping.get(self._current_level, "LOW")
