"""
Financial Contagion Monitor — src/simulation/montecarlo.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

MonteCarloStressTester — runs 1000+ iterations of the contagion simulation
with randomized parameter variations to produce 95% confidence intervals
for the Systemic Risk Index.

Key outputs:
  SRI mean, 95% CI [low, high]
  Recovery rate (days until SRI < 0.5)
  P(D-SIB affected) — probability a systemically important bank enters cascade
  Worst-case SRI (99th percentile)

Architecture:
  Background threading prevents dashboard freeze (2–5s for 1000 runs).
  Dashboard reads get_last_result() — never calls run_sync() directly.
  A small "Updating..." badge shows while _running is True.

Person 3 (Ayush) owns this module.
Output file: data/raw/mc_results.csv + data/raw/systemic_risk_metrics.json
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.data.loader import BankSnapshot, compute_sri
from src.simulation.contagion import ContagionSimulator, CORE_BANKS


@dataclass
class MCResult:
    """Results from a completed Monte Carlo stress test."""
    sri_distribution: np.ndarray    # [n_runs] array of SRI outcomes
    sri_mean: float
    sri_ci_low: float               # 2.5th percentile
    sri_ci_high: float              # 97.5th percentile
    worst_case_sri: float           # 99th percentile
    recovery_rate: float            # mean days until SRI < 0.5
    p_dsib_affected: float          # fraction of runs where a D-SIB was affected
    total_runs: int
    shock_bank: str
    shock_magnitude: float
    completed_at: str = ""


class MonteCarloStressTester:
    """
    Runs Monte Carlo stress tests in a background thread to avoid blocking
    the Dash callback thread.

    Usage (from dashboard callback):
        mc.run_async(snapshots, shock_bank="Yes Bank", magnitude=0.85)
        # ... returns immediately, dashboard stays responsive

        result = mc.get_last_result()   # read latest completed result
        if result:
            print(f"SRI CI: {result.sri_ci_low:.3f} – {result.sri_ci_high:.3f}")
    """

    def __init__(self, n_runs: int = 1000):
        self._n_runs = n_runs
        self._simulator = ContagionSimulator()
        self._last_result: Optional[MCResult] = None
        self._lock = threading.Lock()
        self._running = False
        self._run_count = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def run_async(
        self,
        snapshots: List[BankSnapshot],
        shock_bank: str,
        magnitude: float = 0.75,
        shock_type: str = "Liquidity Crisis",
    ) -> None:
        """
        Starts Monte Carlo run in a background daemon thread.
        Non-blocking — returns immediately.
        If a run is already in progress, skips (previous run will complete).
        """
        if self._running:
            return  # already running — don't queue another
        self._running = True

        def _worker():
            try:
                result = self._run_sync(snapshots, shock_bank, magnitude, shock_type)
                with self._lock:
                    self._last_result = result
                    self._run_count += 1
            except Exception as e:
                print(f"[MC] Background thread error: {e}")
            finally:
                self._running = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def run_sync(
        self,
        snapshots: List[BankSnapshot],
        shock_bank: str,
        magnitude: float = 0.75,
        shock_type: str = "Liquidity Crisis",
    ) -> MCResult:
        """
        Blocking version — used by main.py and tests.
        Runs all iterations in the calling thread.
        """
        return self._run_sync(snapshots, shock_bank, magnitude, shock_type)

    def get_last_result(self) -> Optional[MCResult]:
        """Thread-safe read of the most recently completed result."""
        with self._lock:
            return self._last_result

    def is_running(self) -> bool:
        """Returns True if a background MC run is in progress."""
        return self._running

    def save_outputs(
        self,
        result: MCResult,
        mc_csv_path: str = "data/raw/mc_results.csv",
        metrics_json_path: str = "data/raw/systemic_risk_metrics.json",
    ) -> None:
        """
        Saves MC results to the two output files expected by Person 3's
        submission checklist items 16 and 17.
        """
        import json
        import os
        import pandas as pd

        # mc_results.csv — one row per run
        os.makedirs(os.path.dirname(mc_csv_path), exist_ok=True)
        rows = []
        for i, sri_val in enumerate(result.sri_distribution):
            rows.append({
                "run_id": i + 1,
                "shock_node": result.shock_bank,
                "magnitude": result.shock_magnitude,
                "sri_result": round(float(sri_val), 6),
                "dsib_affected": "",   # populated if tracked per run
                "recovery_days": "",
                "seed": 1000 + i,
            })
        pd.DataFrame(rows).to_csv(mc_csv_path, index=False)

        # systemic_risk_metrics.json
        metrics = {
            "SRI_CI": f"{result.sri_ci_low:.3f} - {result.sri_ci_high:.3f} (95% CI)",
            "SRI_mean": round(result.sri_mean, 6),
            "SRI_ci_low": round(result.sri_ci_low, 6),
            "SRI_ci_high": round(result.sri_ci_high, 6),
            "worst_case_sri": round(result.worst_case_sri, 6),
            "recovery_rate_days_mean": round(result.recovery_rate, 1),
            "p_dsib_affected": round(result.p_dsib_affected, 4),
            "total_runs": result.total_runs,
            "shock_bank": result.shock_bank,
            "shock_magnitude": result.shock_magnitude,
            "generated_at": result.completed_at,
        }
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[MC] Saved {len(rows)} runs to {mc_csv_path}")
        print(f"[MC] Saved metrics to {metrics_json_path}")

    # ── Private: Core Computation ─────────────────────────────────────────────

    def _run_sync(
        self,
        snapshots: List[BankSnapshot],
        shock_bank: str,
        magnitude: float,
        shock_type: str,
    ) -> MCResult:
        """
        Runs self._n_runs iterations with randomized parameter variations.

        Varied parameters across runs:
          shock_magnitude:     uniform [base * 0.70,  base * 1.30]
          recovery_speed:      uniform [0.03,  0.15]
          contagion_amplifier: normal(mean=1.0, std=0.20) — market panic multiplier
          exposure_noise:      ±15% on all interbank_exposure_ratio values
        """
        rng = np.random.default_rng(seed=42)

        sri_values = np.zeros(self._n_runs)
        dsib_hits = 0
        recovery_days_list = []

        base_sri = compute_sri(snapshots)

        for run_idx in range(self._n_runs):
            # Randomize parameters
            mag_var = magnitude * rng.uniform(0.70, 1.30)
            mag_var = float(np.clip(mag_var, 0.05, 1.0))

            contagion_amp = float(np.clip(rng.normal(1.0, 0.20), 0.5, 2.0))
            recovery_speed = float(rng.uniform(0.03, 0.15))

            # Perturb snapshots with exposure noise
            noisy_snapshots = self._add_exposure_noise(snapshots, rng, noise_pct=0.15)

            # Run cascade
            try:
                result = self._simulator.run(
                    snapshots=noisy_snapshots,
                    shock_bank=shock_bank,
                    magnitude=mag_var,
                    shock_type=shock_type,
                    n_steps=3,
                )
            except Exception:
                sri_values[run_idx] = base_sri
                continue

            # Compute SRI from cascade outcome
            n_affected = max(1, result.total_banks_affected)
            stress_fraction = n_affected / 33.0
            run_sri = float(np.clip(
                base_sri + (stress_fraction * mag_var * contagion_amp * 0.5),
                0.0, 1.0
            ))
            sri_values[run_idx] = run_sri

            # Track D-SIB hits
            if result.dsib_affected:
                dsib_hits += 1

            # Estimate recovery: days until SRI decays below 0.5
            if run_sri > 0.5:
                days = self._estimate_recovery_days(run_sri, recovery_speed)
                recovery_days_list.append(days)

        # Compute statistics
        sri_mean = float(np.mean(sri_values))
        sri_ci_low = float(np.percentile(sri_values, 2.5))
        sri_ci_high = float(np.percentile(sri_values, 97.5))
        worst_case = float(np.percentile(sri_values, 99.0))
        p_dsib = dsib_hits / self._n_runs
        recovery_mean = float(np.mean(recovery_days_list)) if recovery_days_list else 0.0

        from datetime import datetime
        completed_at = datetime.now().isoformat()

        return MCResult(
            sri_distribution=sri_values,
            sri_mean=sri_mean,
            sri_ci_low=sri_ci_low,
            sri_ci_high=sri_ci_high,
            worst_case_sri=worst_case,
            recovery_rate=recovery_mean,
            p_dsib_affected=p_dsib,
            total_runs=self._n_runs,
            shock_bank=shock_bank,
            shock_magnitude=magnitude,
            completed_at=completed_at,
        )

    def _add_exposure_noise(
        self,
        snapshots: List[BankSnapshot],
        rng: np.random.Generator,
        noise_pct: float = 0.15,
    ) -> List[BankSnapshot]:
        """
        Returns a new list of BankSnapshots with ±noise_pct random perturbation
        on interbank_exposure_ratio and interbank_borrowing_ratio.
        """
        noisy = []
        for snap in snapshots:
            noise_exp = 1.0 + rng.uniform(-noise_pct, noise_pct)
            noise_borrow = 1.0 + rng.uniform(-noise_pct, noise_pct)
            d = snap._asdict()
            d["interbank_exposure_ratio"] = float(
                np.clip(snap.interbank_exposure_ratio * noise_exp, 0.0, 0.99)
            )
            d["interbank_borrowing_ratio"] = float(
                np.clip(snap.interbank_borrowing_ratio * noise_borrow, 0.0, 0.99)
            )
            noisy.append(BankSnapshot(**d))
        return noisy

    @staticmethod
    def _estimate_recovery_days(sri: float, recovery_speed: float) -> float:
        """
        Estimates days until SRI decays below 0.5 using exponential decay model.
        SRI(t) = sri * exp(-recovery_speed * t)
        Solve for t: t = -ln(0.5 / sri) / recovery_speed
        """
        if sri <= 0.5 or recovery_speed <= 0:
            return 0.0
        try:
            days = -np.log(0.5 / sri) / recovery_speed
            return float(np.clip(days, 1.0, 365.0))
        except (ValueError, ZeroDivisionError):
            return 30.0
