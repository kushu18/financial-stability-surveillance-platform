"""
Financial Contagion Monitor — src/gnn/explainer.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

GNNBankExplainer — provides per-bank feature importance for the dashboard
and LLM report generator.

For any given bank, explains WHICH of the 14 input features contributed most
to the GNN's vulnerability prediction. Uses gradient-based explanation
(input × gradient) — a standard, lightweight explainability method that
doesn't require installing additional XAI libraries.

Critical contract:
  explain_bank("Yes Bank").risk_level    == "CRITICAL"
  explain_bank("Yes Bank").top_features[0][0] == "npl_ratio"
  (Yes Bank's NPA spiral is the real historical reason for its distress)

Used by:
  ReportGenerator — section 4 "Banks Likely To Be Infected Next"
  Dashboard callbacks — bank stats panel
  Panel Q&A — "how do we know the GNN works?"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.gnn.inference import GNNInferenceEngine, FEATURE_ORDER


@dataclass
class ExplainerResult:
    """
    Complete explanation for one bank's GNN vulnerability prediction.
    Produced by GNNBankExplainer.explain_bank().
    Consumed by ReportGenerator and dashboard callbacks.
    """
    bank_name: str
    risk_level: str                          # "SAFE"/"LOW"/"MODERATE"/"HIGH"/"CRITICAL"
    vulnerability_score: float               # P(vulnerable) from GNN [0, 1]
    top_features: List[Tuple[str, float]]    # [(feature_name, importance), ...] sorted desc
    explanation_text: str                    # Human-readable summary for report
    all_importances: Dict[str, float] = field(default_factory=dict)  # all 14 features


class GNNBankExplainer:
    """
    Wraps GNNInferenceEngine to provide human-readable feature importance
    explanations for individual banks.

    Usage:
        explainer = GNNBankExplainer(gnn_engine)
        result = explainer.explain_bank("Yes Bank", day_result.snapshots)
        print(result.risk_level)           # "CRITICAL"
        print(result.top_features[:3])     # [("npl_ratio", 0.42), ...]
        print(result.explanation_text)     # Paragraph for LLM report
    """

    # Human-readable descriptions of each feature for the explanation text
    FEATURE_DESCRIPTIONS = {
        "tier1_capital_ratio":       "core equity capital adequacy",
        "total_capital_ratio":       "total regulatory capital buffer",
        "leverage_ratio":            "systemic leverage exposure",
        "lcr":                       "short-term liquidity coverage",
        "nsfr":                      "long-term funding stability",
        "npl_ratio":                 "non-performing loan ratio",
        "cds_spread_bps":            "credit default swap spread (market distress signal)",
        "equity_volatility_30d":     "30-day equity market volatility",
        "interbank_borrowing_ratio": "interbank borrowing dependency",
        "interbank_exposure_ratio":  "interbank lending exposure",
        "network_degree":            "interbank network connectivity",
        "eigenvector_centrality":    "systemic network importance",
        "gsib_score":                "global systemic importance score",
        "capital_buffer":            "regulatory capital cushion",
    }

    # Risk level thresholds — must match GNNInferenceEngine.get_risk_level()
    RISK_THRESHOLDS = {
        "CRITICAL": 0.80,
        "HIGH":     0.60,
        "MODERATE": 0.40,
        "LOW":      0.20,
        "SAFE":     0.00,
    }

    def __init__(self, gnn_engine: GNNInferenceEngine):
        """
        Args:
            gnn_engine: Initialized GNNInferenceEngine (or MockGNNInferenceEngine)
        """
        self._engine = gnn_engine

    def explain_bank(
        self,
        bank_name: str,
        snapshots: list,
        n_top_features: int = 5
    ) -> ExplainerResult:
        """
        Generates a complete explanation for one bank's vulnerability prediction.

        Args:
            bank_name:       Full institution_id (e.g. "Yes Bank")
            snapshots:       List of 33 BankSnapshot objects (from DayResult)
            n_top_features:  How many top features to include (default 5)

        Returns:
            ExplainerResult with risk level, top features, and prose explanation
        """
        # Get vulnerability probability
        vuln_prob = self._engine.predict_single(bank_name, snapshots)
        risk_level = self._engine.get_risk_level(vuln_prob)

        # Get feature importances via gradient method
        importances = self._engine.compute_feature_importance(bank_name, snapshots)
        top_features = importances[:n_top_features]

        # All importances as dict
        all_importances = {feat: score for feat, score in importances}

        # Build explanation text
        explanation_text = self._build_explanation_text(
            bank_name, risk_level, vuln_prob, top_features, snapshots
        )

        return ExplainerResult(
            bank_name=bank_name,
            risk_level=risk_level,
            vulnerability_score=vuln_prob,
            top_features=top_features,
            explanation_text=explanation_text,
            all_importances=all_importances,
        )

    def explain_all_banks(
        self,
        snapshots: list,
        n_top_features: int = 3
    ) -> Dict[str, ExplainerResult]:
        """
        Explains all 33 banks. Used for report generation and dashboard tables.

        Returns:
            Dict of {bank_name: ExplainerResult}
        """
        all_probs = self._engine.predict(snapshots)
        results = {}

        for snap in snapshots:
            bank_name = snap.institution_id
            vuln_prob = all_probs.get(bank_name, 0.5)
            risk_level = self._engine.get_risk_level(vuln_prob)

            # Compute importances for at-risk banks only (performance)
            if vuln_prob > 0.3:
                importances = self._engine.compute_feature_importance(
                    bank_name, snapshots
                )
            else:
                # Safe banks: skip expensive gradient computation
                importances = [(f, 0.0) for f in FEATURE_ORDER]

            top_features = importances[:n_top_features]
            all_importances = {feat: score for feat, score in importances}

            explanation_text = self._build_explanation_text(
                bank_name, risk_level, vuln_prob, top_features, snapshots
            )

            results[bank_name] = ExplainerResult(
                bank_name=bank_name,
                risk_level=risk_level,
                vulnerability_score=vuln_prob,
                top_features=top_features,
                explanation_text=explanation_text,
                all_importances=all_importances,
            )

        return results

    def get_top_at_risk_banks(
        self,
        snapshots: list,
        n: int = 5
    ) -> List[ExplainerResult]:
        """
        Returns the top N most vulnerable banks with full explanations.
        Used by ReportGenerator section 4.
        """
        all_results = self.explain_all_banks(snapshots)
        sorted_results = sorted(
            all_results.values(),
            key=lambda r: r.vulnerability_score,
            reverse=True
        )
        return sorted_results[:n]

    # ── Private: Explanation Text Builder ────────────────────────────────────

    def _build_explanation_text(
        self,
        bank_name: str,
        risk_level: str,
        vuln_prob: float,
        top_features: List[Tuple[str, float]],
        snapshots: list,
    ) -> str:
        """
        Builds a 2–3 sentence human-readable explanation of why a bank
        has its current vulnerability score. Used in the LLM report and
        dashboard bank stats panel.
        """
        # Get the bank snapshot for specific metric values
        snap = next((s for s in snapshots if s.institution_id == bank_name), None)

        # Intro sentence
        if risk_level in ("CRITICAL", "HIGH"):
            intro = (
                f"{bank_name} is assessed as {risk_level} with a GNN vulnerability "
                f"probability of {vuln_prob:.1%}."
            )
        elif risk_level == "MODERATE":
            intro = (
                f"{bank_name} shows MODERATE vulnerability ({vuln_prob:.1%}) "
                f"requiring monitoring."
            )
        else:
            intro = (
                f"{bank_name} is assessed as {risk_level} "
                f"(vulnerability: {vuln_prob:.1%})."
            )

        # Top driver sentence
        if top_features and top_features[0][1] > 0.01:
            top_feat_name = top_features[0][0]
            top_feat_desc = self.FEATURE_DESCRIPTIONS.get(
                top_feat_name, top_feat_name.replace("_", " ")
            )
            # Get actual metric value from snapshot
            metric_val = ""
            if snap:
                val = getattr(snap, top_feat_name, None)
                if val is not None:
                    metric_val = f" (current value: {val:.3f})"

            drivers = (
                f"The primary driver is {top_feat_desc}{metric_val}."
            )

            # Secondary driver if available
            if len(top_features) > 1 and top_features[1][1] > 0.01:
                sec_feat_name = top_features[1][0]
                sec_feat_desc = self.FEATURE_DESCRIPTIONS.get(
                    sec_feat_name, sec_feat_name.replace("_", " ")
                )
                drivers += f" Secondary factor: {sec_feat_desc}."
        else:
            drivers = "Feature importance analysis is inconclusive for this date."

        # Contagion channel sentence for high-risk banks
        contagion = ""
        if risk_level in ("CRITICAL", "HIGH") and snap:
            if snap.interbank_borrowing_ratio > 0.15:
                contagion = (
                    f"High interbank borrowing dependency "
                    f"({snap.interbank_borrowing_ratio:.1%}) "
                    f"creates contagion transmission risk to connected banks."
                )
            elif snap.interbank_exposure_ratio > 0.12:
                contagion = (
                    f"Significant interbank exposure "
                    f"({snap.interbank_exposure_ratio:.1%}) "
                    f"may propagate stress to lending counterparties."
                )

        parts = [intro, drivers]
        if contagion:
            parts.append(contagion)

        return " ".join(parts)
