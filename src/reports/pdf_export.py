"""
Financial Contagion Monitor — src/reports/pdf_export.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

PDFExporter — generates a professionally formatted A4 PDF from a ReportResult.

Layout:
  - Black header bar with project name and bank name
  - Crisis severity badge (color-coded by EWS level)
  - 6 report sections with proper typography
  - Risk metrics summary table (SRI, CI, top banks)
  - Contagion chain table (if available)
  - Footer with generation timestamp

Output: reports/exports/FCM_Report_{date}_{bank_short}.pdf
"""

import os
from datetime import datetime
from typing import Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable, PageBreak,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from src.reports.generator import ReportResult
from src.data.loader import DayResult

# Colour palette — matches dashboard black theme
BLACK       = colors.HexColor("#050508") if REPORTLAB_AVAILABLE else None
DARK_BG     = colors.HexColor("#0d1117") if REPORTLAB_AVAILABLE else None
ACCENT_BLUE = colors.HexColor("#0057FF") if REPORTLAB_AVAILABLE else None
CYAN        = colors.HexColor("#00C2FF") if REPORTLAB_AVAILABLE else None
RED         = colors.HexColor("#F85149") if REPORTLAB_AVAILABLE else None
AMBER       = colors.HexColor("#D29922") if REPORTLAB_AVAILABLE else None
GREEN       = colors.HexColor("#3FB950") if REPORTLAB_AVAILABLE else None
LIGHT_TEXT  = colors.HexColor("#E6EDF3") if REPORTLAB_AVAILABLE else None
MUTED_TEXT  = colors.HexColor("#8B949E") if REPORTLAB_AVAILABLE else None
BORDER      = colors.HexColor("#21262D") if REPORTLAB_AVAILABLE else None

EWS_COLORS = {
    "NORMAL":   colors.HexColor("#3FB950") if REPORTLAB_AVAILABLE else None,
    "ELEVATED": colors.HexColor("#D29922") if REPORTLAB_AVAILABLE else None,
    "CRITICAL": colors.HexColor("#F85149") if REPORTLAB_AVAILABLE else None,
}

OUTPUT_DIR = "reports/exports"


class PDFExporter:
    """
    Exports a ReportResult to a professionally formatted PDF.

    Usage:
        exporter = PDFExporter()
        path = exporter.export(report_result, day_result)
        # path = "reports/exports/FCM_Report_2020-03-15_SBI.pdf"
    """

    def export(
        self,
        report: ReportResult,
        day_result: DayResult,
        output_path: str = None,
    ) -> str:
        """
        Generates and saves the PDF report.

        Args:
            report:      ReportResult from ReportGenerator
            day_result:  DayResult providing context (date, bank info, etc.)
            output_path: Override output path (optional)

        Returns:
            Path to the saved PDF file
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab is required for PDF export: pip install reportlab"
            )

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Determine bank short name for filename
        bank_short = "FCM"
        if day_result.snapshots:
            # Use the first snapshot's bank info (or the top risk bank)
            bank_short = day_result.snapshots[0].bank_short or "FCM"

        # Build output path
        if output_path is None:
            safe_date = day_result.date.replace("/", "-").replace(":", "-").replace(" ", "_")
            filename = f"FCM_Report_{safe_date}_{bank_short}.pdf"
            output_path = os.path.join(OUTPUT_DIR, filename)

        # Build PDF
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=15 * mm,
            bottomMargin=20 * mm,
        )

        styles = self._build_styles()
        story = self._build_story(report, day_result, styles)

        doc.build(
            story,
            onFirstPage=self._draw_header_footer,
            onLaterPages=self._draw_header_footer,
        )

        return output_path

    # ── Private: Story Builder ────────────────────────────────────────────────

    def _build_story(self, report: ReportResult, day_result: DayResult, styles: dict) -> list:
        """Assembles the reportlab Platypus story."""
        story = []
        data = report.data_snapshot

        # ── Title Block ──
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph(
            "FINANCIAL CONTAGION MONITOR",
            styles["title"]
        ))
        story.append(Paragraph(
            "Crisis Analysis Report — Reserve Bank of India",
            styles["subtitle"]
        ))
        story.append(Spacer(1, 4 * mm))
        story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE))
        story.append(Spacer(1, 3 * mm))

        # ── Metadata Row ──
        meta_data = [
            ["Date", day_result.date],
            ["Mode", data.get("data_mode", "historical").upper()],
            ["EWS Level", data.get("ews_level", "NORMAL")],
            ["Active Crisis", data.get("crisis_name", "None")],
            ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")],
            ["Report Engine", f"{report.generated_by.upper()} ({report.model_used or 'template'})"],
        ]
        meta_table = Table(meta_data, colWidths=[45 * mm, 120 * mm])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (0, -1), DARK_BG),
            ("TEXTCOLOR",    (0, 0), (0, -1), MUTED_TEXT),
            ("TEXTCOLOR",    (1, 0), (1, -1), LIGHT_TEXT),
            ("FONTNAME",     (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BLACK, DARK_BG]),
            ("GRID",         (0, 0), (-1, -1), 0.5, BORDER),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 5 * mm))

        # ── SRI Metrics Table ──
        story.append(Paragraph("SYSTEMIC RISK METRICS", styles["section_header"]))
        story.append(Spacer(1, 2 * mm))

        sri = data.get("sri", 0)
        sri_color = RED if sri > 0.65 else (AMBER if sri > 0.45 else GREEN)

        metrics_data = [
            ["Metric", "Value", "Threshold", "Status"],
            ["Systemic Risk Index (SRI)",
             f"{sri:.4f}",
             "< 0.45 Normal",
             "CRITICAL" if sri > 0.65 else "ELEVATED" if sri > 0.45 else "NORMAL"],
            ["SRI 95% CI Low",    f"{data.get('sri_ci_low', sri):.4f}", "—", "—"],
            ["SRI 95% CI High",   f"{data.get('sri_ci_high', sri):.4f}", "—", "—"],
            ["Worst-Case SRI",    f"{data.get('worst_case_sri', sri):.4f}", "< 0.80", "—"],
            ["Banks Stressed",    f"{data.get('n_banks_stressed', 0)}/33", "< 5", "—"],
            ["Banks Critical",    f"{data.get('n_banks_critical', 0)}/33", "0", "—"],
            ["P(D-SIB Affected)", f"{data.get('p_dsib_affected', 0):.1%}", "< 5%", "—"],
            ["Recovery Estimate", f"{data.get('recovery_days', 0):.0f} days", "< 30 days", "—"],
        ]

        metrics_table = Table(
            metrics_data,
            colWidths=[70 * mm, 35 * mm, 40 * mm, 30 * mm]
        )
        metrics_table.setStyle(TableStyle([
            # Header row
            ("BACKGROUND",   (0, 0), (-1, 0), ACCENT_BLUE),
            ("TEXTCOLOR",    (0, 0), (-1, 0), LIGHT_TEXT),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0), 9),
            # Data rows
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BLACK, DARK_BG]),
            ("TEXTCOLOR",    (0, 1), (-1, -1), LIGHT_TEXT),
            ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",     (0, 1), (-1, -1), 8),
            ("GRID",         (0, 0), (-1, -1), 0.5, BORDER),
            ("LEFTPADDING",  (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            # Highlight SRI row
            ("TEXTCOLOR",    (1, 1), (1, 1), sri_color),
            ("FONTNAME",     (1, 1), (1, 1), "Helvetica-Bold"),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 5 * mm))

        # ── Top At-Risk Banks Table ──
        top_risks = data.get("top_risks", [])
        if top_risks:
            story.append(Paragraph("TOP AT-RISK BANKS", styles["section_header"]))
            story.append(Spacer(1, 2 * mm))
            risk_rows = [["Bank", "Short", "Vulnerability", "NPL Ratio", "CDS (bps)", "Tier 1"]]
            for r in top_risks[:5]:
                vuln = r.get("vuln", 0)
                v_color_str = "CRITICAL" if vuln > 0.80 else "HIGH" if vuln > 0.60 else "MOD"
                risk_rows.append([
                    r.get("bank", ""),
                    r.get("short", ""),
                    f"{vuln:.4f} [{v_color_str}]",
                    f"{r.get('npl', 0):.4f}",
                    f"{r.get('cds', 0):.0f}",
                    f"{r.get('tier1', 0):.4f}",
                ])
            risk_table = Table(
                risk_rows,
                colWidths=[55 * mm, 18 * mm, 32 * mm, 22 * mm, 22 * mm, 22 * mm]
            )
            risk_table.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0), ACCENT_BLUE),
                ("TEXTCOLOR",    (0, 0), (-1, 0), LIGHT_TEXT),
                ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, 0), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BLACK, DARK_BG]),
                ("TEXTCOLOR",    (0, 1), (-1, -1), LIGHT_TEXT),
                ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",     (0, 1), (-1, -1), 7.5),
                ("GRID",         (0, 0), (-1, -1), 0.5, BORDER),
                ("LEFTPADDING",  (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING",   (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 5 * mm))

        # ── Contagion Chain Table ──
        cascade_steps = data.get("cascade_steps", [])
        if cascade_steps:
            story.append(Paragraph("CONTAGION CASCADE CHAIN", styles["section_header"]))
            story.append(Spacer(1, 2 * mm))
            cascade_rows = [["Step", "Bank", "Channel", "Exposure", "Vulnerability", "Status"]]
            for s in cascade_steps[:8]:
                cascade_rows.append([
                    f"T+{s['step']}" if s["step"] > 0 else "ORIGIN",
                    s.get("bank", ""),
                    s.get("channel", ""),
                    f"{s.get('exposure', 0):.3f}",
                    f"{s.get('vuln', 0):.4f}",
                    s.get("status", ""),
                ])
            cascade_table = Table(
                cascade_rows,
                colWidths=[16 * mm, 55 * mm, 22 * mm, 20 * mm, 26 * mm, 22 * mm]
            )
            cascade_table.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#5c0000")),
                ("TEXTCOLOR",    (0, 0), (-1, 0), LIGHT_TEXT),
                ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, 0), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BLACK, DARK_BG]),
                ("TEXTCOLOR",    (0, 1), (-1, -1), LIGHT_TEXT),
                ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",     (0, 1), (-1, -1), 7.5),
                ("GRID",         (0, 0), (-1, -1), 0.5, BORDER),
                ("LEFTPADDING",  (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING",   (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ]))
            story.append(cascade_table)
            story.append(Spacer(1, 5 * mm))

            # RBI intervention banner
            if data.get("rbi_intervention"):
                injection = data.get("injection_cr", 0)
                story.append(Paragraph(
                    f"⚠ RBI INTERVENTION REQUIRED — D-SIB AFFECTED | "
                    f"ESTIMATED INJECTION: Rs {injection:,.0f} CRORE",
                    styles["warning_banner"]
                ))
                story.append(Spacer(1, 4 * mm))

        # ── Report Sections ──
        story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("ANALYSIS REPORT", styles["section_header"]))
        story.append(Spacer(1, 3 * mm))

        section_order = [
            "1. Crisis Executive Summary",
            "2. Systemic Risk Index Analysis",
            "3. Contagion Chain Analysis",
            "4. Banks Likely to Be Infected Next",
            "5. RBI Action Required?",
            "6. Recovery Timeline Estimate",
        ]

        for section_title in section_order:
            body = report.sections.get(section_title, "")
            if not body:
                continue
            story.append(Paragraph(section_title, styles["subsection_header"]))
            story.append(Spacer(1, 1 * mm))
            # Split into paragraphs
            for para in body.split("\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(para, styles["body_text"]))
                    story.append(Spacer(1, 1.5 * mm))
            story.append(Spacer(1, 3 * mm))

        return story

    # ── Private: Styles ───────────────────────────────────────────────────────

    def _build_styles(self) -> dict:
        """Builds reportlab paragraph styles matching the dark dashboard theme."""
        base = getSampleStyleSheet()

        def ps(name, **kwargs):
            return ParagraphStyle(name, **kwargs)

        return {
            "title": ps("fcm_title",
                fontName="Helvetica-Bold", fontSize=16,
                textColor=LIGHT_TEXT, backColor=BLACK,
                alignment=TA_CENTER, spaceAfter=4),

            "subtitle": ps("fcm_subtitle",
                fontName="Helvetica", fontSize=10,
                textColor=MUTED_TEXT, backColor=BLACK,
                alignment=TA_CENTER, spaceAfter=2),

            "section_header": ps("fcm_section",
                fontName="Helvetica-Bold", fontSize=9,
                textColor=CYAN, backColor=BLACK,
                spaceAfter=2, spaceBefore=4,
                letterSpacing=1.5),

            "subsection_header": ps("fcm_subsection",
                fontName="Helvetica-Bold", fontSize=9,
                textColor=ACCENT_BLUE, backColor=BLACK,
                spaceAfter=1, spaceBefore=2),

            "body_text": ps("fcm_body",
                fontName="Helvetica", fontSize=8.5,
                textColor=LIGHT_TEXT, backColor=BLACK,
                leading=13, spaceAfter=0),

            "warning_banner": ps("fcm_warning",
                fontName="Helvetica-Bold", fontSize=9,
                textColor=LIGHT_TEXT,
                backColor=colors.HexColor("#5c0000"),
                alignment=TA_CENTER,
                borderPadding=6, spaceAfter=2),
        }

    # ── Private: Header/Footer ────────────────────────────────────────────────

    def _draw_header_footer(self, canvas, doc) -> None:
        """Draws the black header bar and footer on every page."""
        canvas.saveState()
        w, h = A4

        # Header bar
        canvas.setFillColor(BLACK)
        canvas.rect(0, h - 12 * mm, w, 12 * mm, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(LIGHT_TEXT)
        canvas.drawString(20 * mm, h - 8 * mm, "FINANCIAL CONTAGION MONITOR")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED_TEXT)
        canvas.drawRightString(
            w - 20 * mm, h - 8 * mm,
            "M.S. Ramaiah Institute of Technology | AIP67 | CONFIDENTIAL"
        )

        # Blue accent line under header
        canvas.setStrokeColor(ACCENT_BLUE)
        canvas.setLineWidth(1.5)
        canvas.line(0, h - 12 * mm, w, h - 12 * mm)

        # Footer
        canvas.setFillColor(DARK_BG)
        canvas.rect(0, 0, w, 12 * mm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED_TEXT)
        canvas.drawString(
            20 * mm, 5 * mm,
            f"Generated by Financial Contagion Monitor | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Page {doc.page}"
        )
        canvas.drawRightString(
            w - 20 * mm, 5 * mm,
            "CONFIDENTIAL — FOR AUTHORISED RBI USE ONLY"
        )

        canvas.restoreState()
