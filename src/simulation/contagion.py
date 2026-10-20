"""
Financial Contagion Monitor — src/simulation/contagion.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

ContagionSimulator — directed cascade model where financial stress propagates
from a shocked bank to its network neighbors based on exposure weights and
current vulnerability states.

Primary propagation channels:
  interbank_borrowing_ratio  — who borrowed from whom (borrowing side)
  interbank_exposure_ratio   — who is exposed to whom (lending side)

The graph is rebuilt from current DayResult snapshots on every run,
NOT from graph.pt. This ensures simulation uses live/current exposure values.

Used by:
  dashboard/callbacks.py  — shock simulation playground (Feature 3)
  src/reports/generator.py — contagion chain section of LLM report
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

from src.data.loader import BankSnapshot

# 5 Core D-SIBs — any appearance triggers RBI intervention flag
CORE_BANKS = [
    "State Bank of India (SBI)",
    "HDFC Bank",
    "ICICI Bank",
    "Axis Bank",
    "Kotak Mahindra Bank",
]

# USD/INR conversion for liquidity injection estimate
USD_TO_INR = 82.0
# 1 USD Bn = 8200 Cr INR
BN_USD_TO_CR_INR = USD_TO_INR * 100.0


@dataclass
class CascadeStep:
    """One propagation step in the contagion cascade."""
    step: int                    # 0 = origin, 1 = first wave, etc.
    bank_name: str
    bank_short: str
    via_channel: str             # "ORIGIN" / "LIQUIDITY" / "CAPITAL" / "CREDIT" / "MARKET"
    exposure_weight: float       # edge weight that transmitted the shock [0, 1]
    resulting_vulnerability: float  # updated vulnerability after shock [0, 1]
    status: str                  # "DISTRESS" / "STRESSED" / "ELEVATED"


@dataclass
class CascadeResult:
    """Complete result of one ContagionSimulator.run() call."""
    origin_bank: str
    shock_type: str
    magnitude: float
    steps: List[CascadeStep] = field(default_factory=list)
    dsib_affected: List[str] = field(default_factory=list)
    rbi_intervention_required: bool = False
    estimated_injection_cr: float = 0.0   # Rs Crore
    total_banks_affected: int = 0
    simulation_date: str = ""


class ContagionSimulator:
    """
    Simulates financial contagion cascades through the interbank network.

    Usage:
        simulator = ContagionSimulator()
        result = simulator.run(
            snapshots=day_result.snapshots,
            shock_bank="Yes Bank",
            magnitude=0.85,
            shock_type="Liquidity Crisis",
            n_steps=3
        )
    """

    # Shock type → feature perturbation map
    SHOCK_TYPES = {
        "Liquidity Crisis": {
            "lcr": -0.40,
            "nsfr": -0.30,
            "short_term_funding_ratio": +0.25,
        },
        "Capital Shortfall": {
            "tier1_capital_ratio": -0.30,
            "capital_buffer": -0.40,
            "total_capital_ratio": -0.25,
        },
        "NPL Spike": {
            "npl_ratio": +0.30,
            "loan_loss_provision_ratio": +0.20,
            "roe": -0.30,
        },
        "Market Panic": {
            "cds_spread_bps": 3.0,     # multiplier, not delta
            "equity_volatility_30d": 2.5,
            "beta_to_financials": +0.40,
        },
    }

    # Primary transmission channel label per shock type
    CHANNEL_LABELS = {
        "Liquidity Crisis":  "LIQUIDITY",
        "Capital Shortfall": "CAPITAL",
        "NPL Spike":         "CREDIT",
        "Market Panic":      "MARKET",
    }

    def __init__(self, node_mapping_path: str = "data/raw/node_mapping.csv"):
        self._node_mapping_path = node_mapping_path
        self.last_result: Optional[CascadeResult] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        snapshots: List[BankSnapshot],
        shock_bank: str,
        magnitude: float,
        shock_type: str = "Liquidity Crisis",
        n_steps: int = 3,
    ) -> CascadeResult:
        """
        Runs the contagion cascade simulation.

        Args:
            snapshots:  Current 33 BankSnapshots (from DayResult)
            shock_bank: Full institution_id of the origin bank
            magnitude:  Shock magnitude [0.10, 1.00]
            shock_type: One of SHOCK_TYPES keys
            n_steps:    Number of propagation steps (1–5)

        Returns:
            CascadeResult with full cascade chain and intervention flags
        """
        magnitude = float(np.clip(magnitude, 0.0, 1.0))
        n_steps = int(np.clip(n_steps, 1, 5))

        if shock_type not in self.SHOCK_TYPES:
            shock_type = "Liquidity Crisis"

        # Build working copy of snapshot dict (mutable)
        snap_dict: Dict[str, BankSnapshot] = {
            s.institution_id: s for s in snapshots
        }

        if shock_bank not in snap_dict:
            # Try partial match
            for name in snap_dict:
                if shock_bank.lower() in name.lower():
                    shock_bank = name
                    break

        # Build directed graph from current snapshot values
        G = self._build_digraph(list(snap_dict.values()))

        # Initialize cascade result
        result = CascadeResult(
            origin_bank=shock_bank,
            shock_type=shock_type,
            magnitude=magnitude,
        )

        # Step 0: Apply shock to origin bank
        if shock_bank in snap_dict:
            shocked_snap = self._apply_shock(snap_dict[shock_bank], magnitude, shock_type)
            snap_dict[shock_bank] = shocked_snap

            origin_step = CascadeStep(
                step=0,
                bank_name=shock_bank,
                bank_short=shocked_snap.bank_short,
                via_channel="ORIGIN",
                exposure_weight=magnitude,
                resulting_vulnerability=min(1.0, shocked_snap.vulnerability_label_injected + magnitude * 0.5),
                status="DISTRESS",
            )
            result.steps.append(origin_step)

        # Track which banks have already been affected
        affected_banks = {shock_bank}

        # Propagate cascade step by step
        channel_label = self.CHANNEL_LABELS.get(shock_type, "LIQUIDITY")

        for step_num in range(1, n_steps + 1):
            newly_affected = {}

            # For each bank affected in the previous step, propagate to neighbors
            prev_step_banks = [
                s.bank_name for s in result.steps if s.step == step_num - 1
            ]

            for source_bank in prev_step_banks:
                if source_bank not in snap_dict:
                    continue

                source_snap = snap_dict[source_bank]
                source_vuln = getattr(source_snap, "vulnerability_label_injected", 0.5)

                # Get all banks that source exposes to (outgoing edges)
                if source_bank not in G:
                    continue

                for neighbor in G.successors(source_bank):
                    if neighbor in affected_banks:
                        continue  # already in cascade

                    edge_data = G.get_edge_data(source_bank, neighbor, {})
                    exposure_weight = edge_data.get("weight", 0.1)

                    neighbor_snap = snap_dict.get(neighbor)
                    if neighbor_snap is None:
                        continue

                    # Propagated loss formula:
                    # loss = magnitude × exposure_weight × (1 - neighbor_capital_buffer)
                    capital_buffer = max(0.01, getattr(neighbor_snap, "capital_buffer", 0.05))
                    propagated_loss = (
                        magnitude * exposure_weight * (1.0 - min(capital_buffer, 0.99))
                    )
                    propagated_loss = float(np.clip(propagated_loss, 0.0, 1.0))

                    # Only cascade if propagated loss is significant
                    if propagated_loss < 0.05:
                        continue

                    current_vuln = getattr(neighbor_snap, "vulnerability_label_injected", 0.0)
                    new_vuln = float(np.clip(current_vuln + propagated_loss, 0.0, 1.0))

                    if neighbor not in newly_affected or newly_affected[neighbor][0] < propagated_loss:
                        newly_affected[neighbor] = (propagated_loss, new_vuln, neighbor_snap.bank_short)

            # Add newly affected banks to this step
            for bank_name, (loss, new_vuln, bank_short) in newly_affected.items():
                affected_banks.add(bank_name)

                status = "DISTRESS" if new_vuln > 0.80 else \
                         "STRESSED" if new_vuln > 0.50 else "ELEVATED"

                cascade_step = CascadeStep(
                    step=step_num,
                    bank_name=bank_name,
                    bank_short=bank_short,
                    via_channel=channel_label,
                    exposure_weight=round(loss, 4),
                    resulting_vulnerability=round(new_vuln, 4),
                    status=status,
                )
                result.steps.append(cascade_step)

            if not newly_affected:
                break  # cascade stopped

        # Compute summary fields
        result.total_banks_affected = len(affected_banks)
        result.dsib_affected = [b for b in affected_banks if b in CORE_BANKS]
        result.rbi_intervention_required = len(result.dsib_affected) > 0

        if result.rbi_intervention_required:
            result.estimated_injection_cr = self._estimate_injection(
                result.dsib_affected, snap_dict
            )

        self.last_result = result
        return result

    def get_dsib_exposure(self, cascade: CascadeResult) -> List[str]:
        """Returns list of D-SIBs in the cascade chain."""
        return cascade.dsib_affected

    def to_table_data(self, result: CascadeResult) -> List[dict]:
        """
        Converts CascadeResult to list of dicts for Dash DataTable.
        Columns: step, bank, channel, exposure_weight, vulnerability, status
        """
        rows = []
        for step in result.steps:
            rows.append({
                "Step": f"T+{step.step}" if step.step > 0 else "ORIGIN",
                "Bank": step.bank_name,
                "Short": step.bank_short,
                "Channel": step.via_channel,
                "Exposure": f"{step.exposure_weight:.3f}",
                "Vulnerability": f"{step.resulting_vulnerability:.3f}",
                "Status": step.status,
            })
        return rows

    # ── Private: Shock Application ────────────────────────────────────────────

    def _apply_shock(
        self,
        snapshot: BankSnapshot,
        magnitude: float,
        shock_type: str,
    ) -> BankSnapshot:
        """
        Applies shock perturbations to a bank snapshot.
        Additive perturbations for ratios; multiplicative for market signals.
        Returns a new BankSnapshot (namedtuples are immutable).
        """
        perturbations = self.SHOCK_TYPES.get(shock_type, {})
        snap_dict = snapshot._asdict()

        for field_name, delta in perturbations.items():
            if field_name not in snap_dict:
                continue
            current = float(snap_dict[field_name])
            if field_name in ("cds_spread_bps", "equity_volatility_30d"):
                # Multiplicative
                snap_dict[field_name] = current * (delta * magnitude + (1 - magnitude))
            else:
                # Additive, scaled by magnitude
                snap_dict[field_name] = current + (delta * magnitude)

        # Increase vulnerability based on shock
        shock_vuln_increase = magnitude * 0.6
        current_vuln = float(snap_dict.get("vulnerability_label_injected", 0.0))
        snap_dict["vulnerability_label_injected"] = float(
            np.clip(current_vuln + shock_vuln_increase, 0.0, 1.0)
        )
        snap_dict["distress_event_injected"] = 1

        return BankSnapshot(**snap_dict)

    # ── Private: Graph Construction ───────────────────────────────────────────

    def _build_digraph(self, snapshots: List[BankSnapshot]):
        """
        Builds a directed NetworkX graph from current snapshot exposure values.
        Called fresh on every simulation run to use current data.
        """
        if not NX_AVAILABLE:
            return _SimpleDiGraph(snapshots)

        G = nx.DiGraph()
        snap_dict = {s.institution_id: s for s in snapshots}

        # Add all nodes with attributes
        for s in snapshots:
            G.add_node(
                s.institution_id,
                vulnerability=s.vulnerability_label_injected,
                capital_buffer=s.capital_buffer,
                total_assets=s.total_assets_usd_bn,
                bank_short=s.bank_short,
            )

        # Core ↔ Core: bidirectional high-weight edges
        for i, b1 in enumerate(CORE_BANKS):
            for b2 in CORE_BANKS[i + 1:]:
                if b1 not in snap_dict or b2 not in snap_dict:
                    continue
                w = (snap_dict[b1].interbank_exposure_ratio +
                     snap_dict[b2].interbank_borrowing_ratio) / 2.0
                w = float(np.clip(w, 0.05, 0.50))
                G.add_edge(b1, b2, weight=w)
                G.add_edge(b2, b1, weight=w)

        # Periphery → Core + Core → Periphery
        for s in snapshots:
            if s.institution_id in CORE_BANKS:
                continue
            # Connect to first 2 core banks (SBI and HDFC as primary counterparties)
            for core in CORE_BANKS[:2]:
                if core not in snap_dict:
                    continue
                w_out = float(np.clip(s.interbank_borrowing_ratio, 0.02, 0.30))
                w_in  = float(np.clip(w_out * 0.7, 0.01, 0.20))
                G.add_edge(s.institution_id, core, weight=w_out)
                G.add_edge(core, s.institution_id, weight=w_in)

        return G

    # ── Private: Liquidity Injection Estimate ────────────────────────────────

    def _estimate_injection(
        self,
        dsib_names: List[str],
        snap_dict: Dict[str, BankSnapshot],
    ) -> float:
        """
        Estimates emergency liquidity injection in Rs Crore.

        Two components:
        1. Capital shortfall: max(0, regulatory_min - current_tier1) × total_assets
        2. Stress injection: even if capital is adequate, DSIBs in distress require
           emergency lending. Estimate as 2% of total assets per affected DSIB.
        """
        REGULATORY_MIN_TIER1 = 0.06
        STRESS_INJECTION_RATE = 0.02  # 2% of total assets as emergency liquidity
        total_cr = 0.0
        for name in dsib_names:
            snap = snap_dict.get(name)
            if snap is None:
                continue
            total_assets_cr = snap.total_assets_usd_bn * BN_USD_TO_CR_INR
            # Capital shortfall component
            shortfall_pct = max(0.0, REGULATORY_MIN_TIER1 - snap.tier1_capital_ratio)
            capital_injection = shortfall_pct * total_assets_cr
            # Stress injection component (minimum floor)
            stress_injection = STRESS_INJECTION_RATE * total_assets_cr
            total_cr += max(capital_injection, stress_injection)
        return round(total_cr, 0)


class _SimpleDiGraph:
    """Minimal directed graph fallback when networkx is not installed."""

    def __init__(self, snapshots: List[BankSnapshot]):
        self._edges: Dict[str, List[Tuple[str, float]]] = {}
        snap_dict = {s.institution_id: s for s in snapshots}

        for s in snapshots:
            self._edges.setdefault(s.institution_id, [])

        for i, b1 in enumerate(CORE_BANKS):
            for b2 in CORE_BANKS[i + 1:]:
                if b1 in snap_dict and b2 in snap_dict:
                    w = 0.15
                    self._edges.setdefault(b1, []).append((b2, w))
                    self._edges.setdefault(b2, []).append((b1, w))

        for s in snapshots:
            if s.institution_id not in CORE_BANKS:
                for core in CORE_BANKS[:2]:
                    w = float(np.clip(s.interbank_borrowing_ratio, 0.02, 0.30))
                    self._edges.setdefault(s.institution_id, []).append((core, w))
                    self._edges.setdefault(core, []).append((s.institution_id, w * 0.7))

    def __contains__(self, node): return node in self._edges
    def successors(self, node): return [n for n, _ in self._edges.get(node, [])]
    def get_edge_data(self, src, dst, default=None):
        for n, w in self._edges.get(src, []):
            if n == dst:
                return {"weight": w}
        return default or {}
