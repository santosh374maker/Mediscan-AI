"""
Report Generator — creates a downloadable PDF summary of blood report analysis.
"""
import io
import logging
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

SEVERITY_COLORS = {
    "panic":      (220, 53, 69),    # red
    "critical":   (255, 133, 27),   # orange
    "borderline": (255, 193, 7),    # yellow
    "normal":     (40, 167, 69),    # green
    "unknown":    (108, 117, 125),  # gray
}


def generate_pdf_report(
    username: str,
    filename: str,
    analysis_results: List[dict],
    ai_explanation: str,
    disclaimer: dict,
) -> bytes:
    """
    Generate a PDF summary report using reportlab.
    Returns PDF as bytes for download.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        story = []

        # ── Header ────────────────────────────────────
        title_style = ParagraphStyle("Title", parent=styles["Title"],
                                     fontSize=22, textColor=colors.HexColor("#1a73e8"),
                                     spaceAfter=6)
        story.append(Paragraph("🩺 MediScan AI — Blood Report Analysis", title_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a73e8")))
        story.append(Spacer(1, 0.3*cm))

        meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=10,
                                    textColor=colors.gray)
        story.append(Paragraph(f"Patient: {username} | Report: {filename} | "
                                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
                                meta_style))
        story.append(Spacer(1, 0.5*cm))

        # ── Disclaimer ────────────────────────────────
        disc_color = {
            "emergency": "#dc3545", "critical": "#fd7e14",
            "warning": "#ffc107", "safe": "#28a745"
        }.get(disclaimer.get("level", "safe"), "#6c757d")

        disc_style = ParagraphStyle("Disclaimer", parent=styles["Normal"],
                                    fontSize=9, backColor=colors.HexColor(disc_color),
                                    textColor=colors.white, padding=8, borderRadius=4)
        story.append(Paragraph(disclaimer.get("message", "").replace("\n", "<br/>"), disc_style))
        story.append(Spacer(1, 0.5*cm))

        # ── Results Table ─────────────────────────────
        story.append(Paragraph("Lab Results Summary", styles["Heading2"]))
        story.append(Spacer(1, 0.2*cm))

        table_data = [["Test", "Your Value", "Normal Range", "Unit", "Status"]]
        for r in analysis_results:
            nr = r.get("normal_range", {})
            low = nr.get("min", "")
            high = nr.get("max", "")
            range_str = f"{low} – {high}" if low != "" and high != 9999 else "—"
            table_data.append([
                r["test_name"].title(),
                str(r["value"]),
                range_str,
                r.get("unit", ""),
                r.get("status", ""),
            ])

        col_widths = [4.5*cm, 2.5*cm, 3.5*cm, 2.5*cm, 3*cm]
        table = Table(table_data, colWidths=col_widths, repeatRows=1)

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a73e8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]

        # Color-code status column
        for i, r in enumerate(analysis_results, start=1):
            sev = r.get("severity", "unknown")
            rgb = SEVERITY_COLORS.get(sev, (108, 117, 125))
            style_cmds.append(
                ("TEXTCOLOR", (4, i), (4, i),
                 colors.Color(rgb[0]/255, rgb[1]/255, rgb[2]/255))
            )

        table.setStyle(TableStyle(style_cmds))
        story.append(table)
        story.append(Spacer(1, 0.5*cm))

        # ── AI Explanation ────────────────────────────
        story.append(Paragraph("AI Explanation", styles["Heading2"]))
        story.append(Spacer(1, 0.2*cm))
        body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10,
                                    leading=14, spaceAfter=6)
        for para in ai_explanation.split("\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))

        # ── Specialists ───────────────────────────────
        specialists = disclaimer.get("specialists", [])
        if specialists:
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph("Recommended Specialists", styles["Heading2"]))
            for s in specialists:
                story.append(Paragraph(f"• {s}", body_style))

        # ── Footer disclaimer ─────────────────────────
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.gray))
        footer_style = ParagraphStyle("Footer", parent=styles["Normal"],
                                      fontSize=8, textColor=colors.gray)
        story.append(Paragraph(
            "⚠️ MediScan AI is an educational tool only. This report does not constitute "
            "medical advice, diagnosis, or treatment. Always consult a qualified healthcare "
            "professional before making any health decisions.",
            footer_style
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    except Exception as e:
        logger.error("PDF report generation failed: %s", e)
        return b""
