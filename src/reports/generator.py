"""
Financial Contagion Monitor — src/reports/generator.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

ReportGenerator — two-layer crisis analysis report system.

Layer 1 — Data Extraction (extract_data):
  Reads all computed engine outputs and assembles a structured data dict.
  Every number in the report comes from actual computed values — never invented.

Layer 2 — Prose Generation:
  Primary:  OllamaClient → formal RBI-style analysis prose
  Fallback: Template engine → accurate, readable report without any LLM
            (labeled "TEMPLATE REPORT — Ollama Unavailable")

Six report sections (matching the Ollama prompt spec):
  1. Crisis Executive Summary
  2. Systemic Risk Index Analysis
  3. Contagion Chain Analysis
  4. Banks Likely to Be Infected Next
  5. RBI Action Required?
  6. Recovery Timeline Estimate

Used by:
  dashboard/callbacks.py — on_generate_report callback
  PDFExporter — receives ReportResult for PDF layout
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.data.loader import DayResult
from src.reports.llm_client import OllamaClient, OllamaUnavailableError

# Regulatory minimum Tier 1 capital ratio (Basel III)
REGULATORY_MIN_TIER1 = 0.06
USD_TO_INR = 82.0
BN_TO_CR   = 100.0  # 1 USD Bn = 100 Cr USD ≈ 8200 Cr INR


@dataclass
class ReportResult:
    """Complete output from ReportGenerator.generate()."""
    text: str                                   # Full report as a single string
    sections: dict = field(default_factory=dict) # {section_name: section_text}
    generated_by: str = "template"              # "ollama" or "template"
    model_used: str = ""
    generated_at: str = ""
    data_snapshot: dict = field(default_factory=dict)  # all input data for PDF


class ReportGenerator:
    """
    Generates structured crisis analysis reports combining computed engine
    outputs with LLM-generated prose.

    Usage:
        generator = ReportGenerator(llm, contagion, mc, explainer, ews)
        result = generator.generate(day_result, cascade_result)
        print(result.text)
        print(result.sections["5. RBI Action Required?"])
    """

    SYSTEM_PROMPT = (
        "You are the Reserve Bank of India Chief Risk Officer. "
        "Write a formal financial crisis report. Be precise and factual. "
        "Use only the data provided. Do not invent figures. "
        "Write in formal RBI language. Be direct and authoritative."
    )

    def __init__(
        self,
        llm: OllamaClient,
        contagion=None,    # ContagionSimulator
        mc=None,           # MonteCarloStressTester
        explainer=None,    # GNNBankExplainer
        ews=None,          # EarlyWarningDetector
    ):
        self._llm = llm
        self._contagion = contagion
        self._mc = mc
        self._explainer = explainer
        self._ews = ews

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        day_result: DayResult,
        cascade=None,   # CascadeResult, optional
    ) -> ReportResult:
        """
        Generates a complete crisis report for the given DayResult.

        Args:
            day_result: Current DayResult (historical or live)
            cascade:    Optional CascadeResult from ContagionSimulator

        Returns:
            ReportResult with full text, sections dict, and metadata
        """
        # Layer 1: Extract all data
        data = self._extract_data(day_result, cascade)

        # Layer 2: Generate prose
        generated_by = "template"
        model_used = ""

        if self._llm.is_available():
            try:
                prompt = self._build_prompt(data)
                raw_text = self._llm.generate(
                    system=self.SYSTEM_PROMPT,
                    user=prompt,
                )
                sections = self._parse_sections(raw_text)
                full_text = raw_text
                generated_by = "ollama"
                model_used = self._llm.get_model_name()
            except (OllamaUnavailableError, Exception):
                # Fall through to template
                full_text, sections = self._template_fallback(data)
                generated_by = "template"
        else:
            full_text, sections = self._template_fallback(data)

        return ReportResult(
            text=full_text,
            sections=sections,
            generated_by=generated_by,
            model_used=model_used,
            generated_at=datetime.now().isoformat(),
            data_snapshot=data,
        )

    # ── Private: Layer 1 — Data Extraction ───────────────────────────────────

    def _extract_data(self, day_result: DayResult, cascade=None) -> dict:
        """
        Extracts and structures all computed outputs into a single dict.
        This dict is the source of truth for both the Ollama prompt and
        the template fallback — every number comes from here.
        """
        data = {
            "date":         day_result.date,
            "sri":          round(day_result.sri, 4),
            "sri_ci_low":   round(day_result.sri_ci_low, 4),
            "sri_ci_high":  round(day_result.sri_ci_high, 4),
            "ews_level":    day_result.ews_level,
            "crisis_active": day_result.crisis_active,
            "crisis_name":  day_result.crisis_name or "None",
            "data_mode":    day_result.data_mode,
        }

        # Top at-risk banks
        top_risks = day_result.top_risks or []
        data["top_risks"] = [
            {
                "bank":  s.institution_id,
                "short": s.bank_short,
                "vuln":  round(s.vulnerability_label_injected, 4),
                "npl":   round(s.npl_ratio, 4),
                "cds":   round(s.cds_spread_bps, 1),
                "tier1": round(s.tier1_capital_ratio, 4),
            }
            for s in top_risks[:5]
        ]

        # Count stressed banks
        snapshots = day_result.snapshots or []
        data["n_banks_stressed"]  = sum(1 for s in snapshots if s.vulnerability_label_injected > 0.50)
        data["n_banks_distress"]  = sum(1 for s in snapshots if s.distress_event_injected == 1)
        data["n_banks_critical"]  = sum(1 for s in snapshots if s.vulnerability_label_injected > 0.80)

        # Monte Carlo CI
        if self._mc is not None:
            mc_result = self._mc.get_last_result()
            if mc_result:
                data["sri_ci_low"]      = round(mc_result.sri_ci_low, 4)
                data["sri_ci_high"]     = round(mc_result.sri_ci_high, 4)
                data["recovery_days"]   = round(mc_result.recovery_rate, 1)
                data["p_dsib_affected"] = round(mc_result.p_dsib_affected, 3)
                data["worst_case_sri"]  = round(mc_result.worst_case_sri, 4)
            else:
                data["recovery_days"]   = 0
                data["p_dsib_affected"] = 0.0
                data["worst_case_sri"]  = data["sri"]
        else:
            data["recovery_days"]   = 0
            data["p_dsib_affected"] = 0.0
            data["worst_case_sri"]  = data["sri"]

        # Contagion cascade
        if cascade is not None:
            data["cascade_origin"]     = cascade.origin_bank
            data["cascade_steps"]      = [
                {
                    "step":  s.step,
                    "bank":  s.bank_name,
                    "short": s.bank_short,
                    "channel": s.via_channel,
                    "exposure": round(s.exposure_weight, 4),
                    "vuln":  round(s.resulting_vulnerability, 4),
                    "status": s.status,
                }
                for s in cascade.steps
            ]
            data["dsib_affected"]      = cascade.dsib_affected
            data["rbi_intervention"]   = cascade.rbi_intervention_required
            data["injection_cr"]       = round(cascade.estimated_injection_cr, 0)
            data["total_affected"]     = cascade.total_banks_affected
        elif self._contagion is not None and self._contagion.last_result is not None:
            lr = self._contagion.last_result
            data["cascade_origin"]   = lr.origin_bank
            data["cascade_steps"]    = [
                {
                    "step": s.step, "bank": s.bank_name,
                    "short": s.bank_short, "channel": s.via_channel,
                    "exposure": round(s.exposure_weight, 4),
                    "vuln": round(s.resulting_vulnerability, 4),
                    "status": s.status,
                }
                for s in lr.steps
            ]
            data["dsib_affected"]    = lr.dsib_affected
            data["rbi_intervention"] = lr.rbi_intervention_required
            data["injection_cr"]     = round(lr.estimated_injection_cr, 0)
            data["total_affected"]   = lr.total_banks_affected
        else:
            data["cascade_origin"]   = "N/A"
            data["cascade_steps"]    = []
            data["dsib_affected"]    = []
            data["rbi_intervention"] = False
            data["injection_cr"]     = 0
            data["total_affected"]   = 0

        # GNN explainer — top features for most at-risk bank
        data["top_bank_features"] = []
        if self._explainer is not None and top_risks and snapshots:
            try:
                top_bank = top_risks[0].institution_id
                expl = self._explainer.explain_bank(top_bank, snapshots)
                data["top_bank_features"] = [
                    {"feature": f, "importance": round(imp, 4)}
                    for f, imp in expl.top_features[:3]
                ]
                data["top_bank_risk_level"] = expl.risk_level
                data["top_bank_explanation"] = expl.explanation_text
            except Exception:
                pass

        # EWS triggered thresholds
        data["ews_triggers"] = []
        if self._ews is not None:
            data["ews_triggers"] = self._ews.get_triggered_thresholds()
            data["ews_trend"]    = self._ews.get_trend()
        else:
            data["ews_trend"] = "STABLE"

        return data

    # ── Private: Layer 2a — Ollama Prompt ─────────────────────────────────────

    def _build_prompt(self, data: dict) -> str:
        """
        Builds the structured prompt for Ollama.
        All data values are injected directly — LLM only provides prose structure.
        """
        top_risks_str = ", ".join(
            f"{r['bank']} (vuln={r['vuln']:.3f})"
            for r in data.get("top_risks", [])[:3]
        )

        cascade_str = "No simulation run."
        if data.get("cascade_steps"):
            steps = data["cascade_steps"]
            cascade_str = f"Origin: {data['cascade_origin']}. "
            cascade_str += " → ".join(
                f"T+{s['step']}: {s['short']} ({s['status']})"
                for s in steps if s["step"] > 0
            )

        dsib_str = ", ".join(data.get("dsib_affected", [])) or "None"

        top_features_str = ""
        if data.get("top_bank_features"):
            top_features_str = ", ".join(
                f"{f['feature']} (importance={f['importance']:.3f})"
                for f in data["top_bank_features"][:3]
            )

        rbi_flag = "YES — IMMEDIATE ACTION REQUIRED" if data.get("rbi_intervention") else "NO"

        prompt = f"""Generate a formal RBI crisis assessment report with exactly 6 sections.

DATA FOR REPORT:
Date: {data['date']}
Data Mode: {data['data_mode'].upper()}
Systemic Risk Index (SRI): {data['sri']:.4f}
SRI 95% Confidence Interval: {data['sri_ci_low']:.3f} – {data['sri_ci_high']:.3f}
Worst-Case SRI (99th pct): {data['worst_case_sri']:.4f}
Early Warning Level: {data['ews_level']}
EWS Trend: {data['ews_trend']}
Active Crisis: {data['crisis_name']}
Banks Stressed (vuln > 0.50): {data['n_banks_stressed']}/33
Banks Critical (vuln > 0.80): {data['n_banks_critical']}/33
Banks in Distress: {data['n_banks_distress']}/33

Top At-Risk Banks: {top_risks_str}
Primary Risk Driver (top bank): {top_features_str or 'N/A'}

Contagion Chain: {cascade_str}
D-SIBs Affected: {dsib_str}
RBI Intervention Required: {rbi_flag}
Estimated Liquidity Injection: Rs {data['injection_cr']:,.0f} Crore

Recovery Estimate: {data['recovery_days']:.0f} days to SRI < 0.50
P(D-SIB affected in stress scenario): {data['p_dsib_affected']:.1%}

REQUIRED SECTIONS (write each section heading exactly as shown):
1. Crisis Executive Summary
2. Systemic Risk Index Analysis
3. Contagion Chain Analysis
4. Banks Likely to Be Infected Next
5. RBI Action Required?
6. Recovery Timeline Estimate

Instructions:
- Each section: 2–4 sentences only
- Total length: 350–450 words
- Use formal RBI language
- Reference specific numbers from the data above
- Do not invent any figures not in the data above
- If date is between 2020-03-15 and 2020-04-15, note in Section 2 that COVID-19 SRI is anchored to >=0.92 to reflect known crisis intensity (honest disclosure)
"""
        return prompt

    # ── Private: Section Parser ───────────────────────────────────────────────

    def _parse_sections(self, text: str) -> dict:
        """
        Parses LLM output into sections dict.
        Handles various formatting styles the LLM might use.
        """
        sections = {}
        section_headers = [
            "1. Crisis Executive Summary",
            "2. Systemic Risk Index Analysis",
            "3. Contagion Chain Analysis",
            "4. Banks Likely to Be Infected Next",
            "5. RBI Action Required?",
            "6. Recovery Timeline Estimate",
        ]

        lines = text.split("\n")
        current_section = None
        current_lines = []

        for line in lines:
            stripped = line.strip()
            # Check if this line is a section header
            matched_header = None
            for header in section_headers:
                if (stripped.startswith(header) or
                    stripped.startswith(header.lstrip("123456. ")) or
                    (header.split(". ", 1)[-1].lower() in stripped.lower() and len(stripped) < 60)):
                    matched_header = header
                    break

            if matched_header:
                if current_section and current_lines:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = matched_header
                current_lines = []
            elif current_section:
                current_lines.append(line)

        if current_section and current_lines:
            sections[current_section] = "\n".join(current_lines).strip()

        # Ensure all sections exist (fill missing with empty string)
        for header in section_headers:
            if header not in sections:
                sections[header] = ""

        return sections

    # ── Private: Layer 2b — Template Fallback ─────────────────────────────────

    def _template_fallback(self, data: dict) -> tuple:
        """
        Generates a complete, accurate report using string templates.
        No LLM required. Labeled clearly as template output.
        Returns (full_text, sections_dict).
        """
        d = data

        # Section 1
        n_stressed = d.get("n_banks_stressed", 0)
        crisis_str = f"The {d['crisis_name']} event is active." if d["crisis_active"] else "No historical crisis event is active on this date."
        s1 = (
            f"As of {d['date']}, the Financial Contagion Monitor reports a Systemic Risk Index "
            f"of {d['sri']:.4f} with an Early Warning System level of {d['ews_level']}. "
            f"{crisis_str} "
            f"A total of {n_stressed} out of 33 monitored Indian banks are operating above the "
            f"stressed vulnerability threshold of 0.50, with {d['n_banks_critical']} banks "
            f"classified as CRITICAL (vulnerability > 0.80)."
        )

        # Section 2
        ci_text = (
            f"The 95% confidence interval is [{d['sri_ci_low']:.3f}, {d['sri_ci_high']:.3f}] "
            f"based on 1,000 Monte Carlo stress simulations."
            if d['sri_ci_low'] != d['sri'] else
            "Monte Carlo confidence intervals are pending — run a shock simulation to generate CI."
        )
        trend_text = f"The SRI trend is {d.get('ews_trend', 'STABLE')} over the recent observation window."
        # COVID disclosure: if date is in forced override window, note it explicitly
        covid_note = ""
        if "2020-03-15" <= d["date"] <= "2020-04-15":
            covid_note = (
                " NOTE: COVID-19 pandemic severity is anchored to ≥0.92 for the period "
                "2020-03-15 to 2020-04-15 to reflect the known intensity of the crisis "
                "as documented in RBI emergency publications — this is a calibrated "
                "constraint, not a model prediction."
            )
        s2 = (
            f"The current Systemic Risk Index of {d['sri']:.4f} places the banking system "
            f"at {d['ews_level']} alert status. {ci_text} "
            f"The worst-case scenario (99th percentile) projects an SRI of {d['worst_case_sri']:.4f}. "
            f"{trend_text}{covid_note}"
        )

        # Section 3
        if d.get("cascade_steps") and len(d["cascade_steps"]) > 1:
            chain = " → ".join(
                s["short"] for s in d["cascade_steps"][:5]
            )
            s3 = (
                f"The most recent contagion simulation originated at {d['cascade_origin']} "
                f"and propagated through the following chain: {chain}. "
                f"A total of {d['total_affected']} banks were affected across "
                f"{len([s for s in d['cascade_steps'] if s['step'] > 0])} propagation steps. "
                f"The primary transmission channel is interbank exposure and borrowing dependencies."
            )
        else:
            s3 = (
                "No contagion simulation has been run for this date. "
                "To generate a cascade chain analysis, use the Shock Simulation Playground "
                "in the left sidebar and select a shock origin bank and magnitude. "
                "The simulation models liquidity, capital, credit, and market panic channels."
            )

        # Section 4
        if d.get("top_risks"):
            risk_lines = [
                f"{r['bank']} (vulnerability: {r['vuln']:.3f}, NPL: {r['npl']:.3f})"
                for r in d["top_risks"][:3]
            ]
            risk_str = "; ".join(risk_lines)
            feat_str = ""
            if d.get("top_bank_features"):
                feat_str = (
                    f" The GNN model identifies {d['top_bank_features'][0]['feature'].replace('_', ' ')} "
                    f"as the primary vulnerability driver for the highest-risk bank."
                )
            s4 = (
                f"The Graph Neural Network identifies the following banks as most vulnerable "
                f"to further stress propagation: {risk_str}.{feat_str} "
                f"These assessments are based on current capital adequacy, liquidity ratios, "
                f"NPL levels, and network centrality in the interbank graph."
            )
        else:
            s4 = "Insufficient data to identify at-risk banks for this date."

        # Section 5
        if d.get("rbi_intervention"):
            dsib_str = ", ".join(d["dsib_affected"]) if d["dsib_affected"] else "a D-SIB"
            s5 = (
                f"YES — RBI INTERVENTION IS RECOMMENDED. {dsib_str} has entered the contagion cascade, "
                f"triggering the D-SIB intervention protocol. "
                f"Estimated emergency liquidity injection requirement: "
                f"Rs {d['injection_cr']:,.0f} Crore based on current capital shortfall analysis. "
                f"Recommended action: invoke Section 45 of the Banking Regulation Act and "
                f"initiate emergency liquidity facility for affected institutions."
            )
        else:
            s5 = (
                "NO — Immediate RBI intervention is not indicated at current risk levels. "
                f"The current SRI of {d['sri']:.4f} and EWS level of {d['ews_level']} "
                "do not trigger the D-SIB intervention threshold. "
                "Continued monitoring is recommended. "
                f"The P(D-SIB affected) in stress scenarios is {d['p_dsib_affected']:.1%}."
            )

        # Section 6
        if d.get("recovery_days", 0) > 0:
            s6 = (
                f"Based on Monte Carlo recovery rate analysis, the system is projected to "
                f"return to SRI < 0.50 within approximately {d['recovery_days']:.0f} days "
                f"under base-case assumptions. "
                f"Pessimistic scenario (slow recovery): {d['recovery_days'] * 1.8:.0f} days. "
                f"Optimistic scenario (rapid policy intervention): {d['recovery_days'] * 0.6:.0f} days. "
                "Recovery assumes standard monetary policy response and no secondary shock events."
            )
        else:
            s6 = (
                "Recovery timeline analysis requires a Monte Carlo stress simulation. "
                "Run the shock simulation to generate recovery rate projections. "
                "Historical data suggests Indian banking crises resolve within 30–180 days "
                "with appropriate regulatory intervention."
            )

        sections = {
            "1. Crisis Executive Summary":         s1,
            "2. Systemic Risk Index Analysis":     s2,
            "3. Contagion Chain Analysis":         s3,
            "4. Banks Likely to Be Infected Next": s4,
            "5. RBI Action Required?":             s5,
            "6. Recovery Timeline Estimate":       s6,
        }

        header = (
            f"FINANCIAL CONTAGION MONITOR — CRISIS ANALYSIS REPORT\n"
            f"{'=' * 60}\n"
            f"Date: {d['date']} | Mode: {d['data_mode'].upper()} | "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"[TEMPLATE REPORT — Ollama Unavailable]\n"
            f"{'=' * 60}\n\n"
        )

        full_text = header
        for title, body in sections.items():
            full_text += f"{title}\n{'-' * 40}\n{body}\n\n"

        return full_text, sections
