# detection/pdf_report.py
"""
PDF Report Generator for SmartClaim Portal.
Produces a professional, multi-section PDF for each claim decision.
"""

import io
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm, mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.graphics.shapes import Drawing, Rect, String
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def _get_status_color(status: str):
    """Return a ReportLab color based on claim status."""
    mapping = {
        "Verified": colors.HexColor("#16A34A"),
        "Rejected": colors.HexColor("#DC2626"),
        "Fraud": colors.HexColor("#7C3AED"),
        "Pending": colors.HexColor("#D97706"),
        "Under Survey": colors.HexColor("#2563EB"),
        "Survey Completed": colors.HexColor("#7C3AED"),
    }
    return mapping.get(status, colors.HexColor("#374151"))


def generate_claim_report_pdf(claim) -> bytes:
    """
    Generate a comprehensive PDF for a given Claim object.

    Args:
        claim: A Claim model instance with all related data.

    Returns:
        bytes: Raw PDF content.
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is not installed. Run: pip install reportlab")

    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=f"SmartClaim Report – {claim.claim_number}",
        author="SmartClaim Portal",
    )

    styles = getSampleStyleSheet()

    # --- Custom Styles ---
    title_style = ParagraphStyle(
        "ClaimTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=2,
    )
    section_heading = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1E40AF"),
        spaceBefore=14,
        spaceAfter=6,
    )
    normal_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#374151"),
        leading=14,
    )
    small_label = ParagraphStyle(
        "SmallLabel",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#6B7280"),
        leading=11,
    )
    bold_style = ParagraphStyle(
        "Bold",
        parent=styles["Normal"],
        fontSize=9,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#111827"),
        leading=14,
    )
    status_style = ParagraphStyle(
        "Status",
        parent=styles["Normal"],
        fontSize=14,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.HexColor("#9CA3AF"),
        alignment=TA_CENTER,
    )

    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    story = []

    # ==========================================
    # HEADER BANNER
    # ==========================================
    header_table = Table(
        [[
            Paragraph("SmartClaim Portal", ParagraphStyle(
                "HeaderTitle", parent=styles["Normal"],
                fontSize=18, fontName="Helvetica-Bold",
                textColor=colors.white,
            )),
            Paragraph(f"Generated: {now}", ParagraphStyle(
                "HeaderDate", parent=styles["Normal"],
                fontSize=8, textColor=colors.HexColor("#CBD5E1"),
                alignment=TA_RIGHT,
            )),
        ]],
        colWidths=["65%", "35%"],
    )
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1E40AF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [6, 6, 6, 6]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # ==========================================
    # CLAIM TITLE & STATUS BADGE
    # ==========================================
    status_color = _get_status_color(claim.status)
    status_cell = Paragraph(
        claim.status.upper(),
        ParagraphStyle("StatusBadge", parent=styles["Normal"],
                       fontSize=11, fontName="Helvetica-Bold",
                       textColor=colors.white, alignment=TA_CENTER),
    )
    title_row = Table(
        [[
            Paragraph(f"Claim Report<br/><font size='11' color='#64748B'>{claim.claim_number}</font>",
                      title_style),
            status_cell,
        ]],
        colWidths=["75%", "25%"],
    )
    title_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (1, 0), (1, 0), status_color),
        ("LEFTPADDING", (1, 0), (1, 0), 8),
        ("RIGHTPADDING", (1, 0), (1, 0), 8),
        ("TOPPADDING", (1, 0), (1, 0), 8),
        ("BOTTOMPADDING", (1, 0), (1, 0), 8),
        ("ROUNDEDCORNERS", (1, 0), (1, 0), [4, 4, 4, 4]),
    ]))
    story.append(title_row)
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=10))

    def info_table(rows, col_widths=None):
        """Helper: render a 2-column label/value table."""
        data = []
        for label_text, value_text in rows:
            data.append([
                Paragraph(label_text, small_label),
                Paragraph(str(value_text) if value_text is not None else "N/A", normal_style),
            ])
        tbl = Table(data, colWidths=col_widths or ["38%", "62%"])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return tbl

    # ==========================================
    # SECTION 1: CLAIM DETAILS
    # ==========================================
    story.append(Paragraph("1. Claim Details", section_heading))
    ph = claim.policyholder
    submitted = claim.submitted_at.strftime("%d %b %Y") if claim.submitted_at else "N/A"
    accident = claim.accident_date.strftime("%d %b %Y") if claim.accident_date else "N/A"
    reviewed_at = claim.reviewed_at.strftime("%d %b %Y, %I:%M %p") if claim.reviewed_at else "Not reviewed yet"

    story.append(info_table([
        ("Claim Number", claim.claim_number),
        ("Policyholder", ph.username if ph else "N/A"),
        ("Email", ph.email if ph else "N/A"),
        ("Accident Date", accident),
        ("Submitted On", submitted),
        ("DL Number", claim.dl_number or "N/A"),
        ("Vehicle Reg No", claim.vehicle_reg_no or "N/A"),
        ("FIR Number", claim.fir_number or "N/A"),
        ("Claim Amount (Original)", f"₹{float(claim.claim_amount):,.2f}"),
        ("Final Claim Amount", f"₹{float(claim.surveyor_assessed_amount or claim.claim_amount):,.2f}"),
        ("Current Status", claim.status),
        ("Reviewed By", claim.reviewed_by or "N/A"),
        ("Reviewed At", reviewed_at),
    ]))

    # Description
    if claim.claim_description:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Description:", bold_style))
        story.append(Paragraph(claim.claim_description, normal_style))

    # ==========================================
    # SECTION 2: AI FRAUD ANALYSIS
    # ==========================================
    story.append(Paragraph("2. AI Fraud Detection Analysis", section_heading))
    fraud_status = "🚨 FRAUD DETECTED" if claim.fraud_detected else "✅ LEGITIMATE"
    story.append(info_table([
        ("Fraud Detection Result", fraud_status),
        ("Confidence Score", f"{float(claim.confidence_score or 0):.1f}%"),
        ("Risk Level", claim.risk_level or "N/A"),
        ("Tabular Fraud Probability", f"{float(claim.tabular_fraud_probability or 0):.1f}%"),
        ("Image Fraud Probability", f"{float(claim.image_fraud_probability or 0):.1f}%"),
        ("Fusion Score", f"{float(claim.fusion_score or 0):.2f}"),
        ("Damage Severity", claim.overall_damage_severity or "N/A"),
        ("Total Damage Areas", str(claim.total_damage_areas or 0)),
        ("Avg Damage %", f"{float(claim.average_damage_percentage or 0):.1f}%"),
        ("Max Damage %", f"{float(claim.max_damage_percentage or 0):.1f}%"),
        ("Images Submitted", str(claim.total_images_submitted or 0)),
        ("DL Verification Score", f"{float(claim.dl_verification_score or 0):.1f}%"),
        ("RTO Verification Score", f"{float(claim.rto_verification_score or 0):.1f}%"),
        ("FIR Verification Score", f"{float(claim.fir_verification_score or 0):.1f}%"),
    ]))

    # ==========================================
    # SECTION 3: SURVEYOR REPORT (if available)
    # ==========================================
    if claim.assigned_surveyor:
        story.append(Paragraph("3. Field Surveyor Report", section_heading))
        survey_done = claim.survey_completed_at.strftime("%d %b %Y, %I:%M %p") if claim.survey_completed_at else "Pending"
        assigned_at = claim.assigned_at.strftime("%d %b %Y") if claim.assigned_at else "N/A"
        story.append(info_table([
            ("Assigned Surveyor", claim.assigned_surveyor.get_full_name() or claim.assigned_surveyor.username),
            ("Assigned On", assigned_at),
            ("Survey Completed", survey_done),
            ("Damage Verified Onsite", "Yes" if claim.damage_verified is True else ("No" if claim.damage_verified is False else "Pending")),
            ("Surveyor Recommendation", claim.surveyor_recommendation or "Pending"),
            ("Surveyor Assessed Amount", f"₹{float(claim.surveyor_assessed_amount):,.2f}" if claim.surveyor_assessed_amount else "Not provided"),
        ]))
        if claim.surveyor_notes:
            story.append(Spacer(1, 8))
            story.append(Paragraph("Field Survey Notes:", bold_style))
            story.append(Paragraph(claim.surveyor_notes, normal_style))

    # ==========================================
    # SECTION 4: ADMIN DECISION
    # ==========================================
    section_num = 4 if claim.assigned_surveyor else 3
    story.append(Paragraph(f"{section_num}. Admin Decision", section_heading))

    decision_color = colors.HexColor("#16A34A") if claim.status == "Verified" else colors.HexColor("#DC2626")
    decision_text = "APPROVED ✅" if claim.status == "Verified" else ("REJECTED ❌" if claim.status == "Rejected" else claim.status.upper())

    decision_table = Table([[
        Paragraph(decision_text, ParagraphStyle(
            "Decision", parent=styles["Normal"],
            fontSize=16, fontName="Helvetica-Bold",
            textColor=colors.white, alignment=TA_CENTER,
        ))
    ]], colWidths=["100%"])
    decision_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), decision_color),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [6, 6, 6, 6]),
    ]))
    story.append(decision_table)

    if claim.admin_notes:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Admin Notes:", bold_style))
        story.append(Paragraph(claim.admin_notes, normal_style))

    # ==========================================
    # SECTION 5: REJECTION REASON (if rejected)
    # ==========================================
    if claim.status == "Rejected" and claim.rejection_reason:
        story.append(Spacer(1, 6))
        rejection_box = Table([[
            Paragraph(
                f"<b>Reason for Rejection:</b><br/>{claim.rejection_reason}",
                ParagraphStyle("RejBox", parent=styles["Normal"],
                               fontSize=9, textColor=colors.HexColor("#7F1D1D"),
                               leading=14),
            )
        ]], colWidths=["100%"])
        rejection_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FECACA")),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(rejection_box)

    # ==========================================
    # SECTION 6: HISTORY TRAIL
    # ==========================================
    history_section = section_num + 1
    story.append(Paragraph(f"{history_section}. Claim History & Audit Trail", section_heading))
    history = list(claim.history.order_by("timestamp"))
    if history:
        hist_data = [
            [
                Paragraph("<b>Timestamp</b>", small_label),
                Paragraph("<b>Action</b>", small_label),
                Paragraph("<b>By</b>", small_label),
                Paragraph("<b>Status Change</b>", small_label),
            ]
        ]
        for entry in history:
            ts = entry.timestamp.strftime("%d %b %y %H:%M")
            status_change = ""
            if entry.old_status and entry.new_status:
                status_change = f"{entry.old_status} → {entry.new_status}"
            hist_data.append([
                Paragraph(ts, small_label),
                Paragraph(entry.action.replace("_", " ").title(), normal_style),
                Paragraph(entry.performed_by or "System", normal_style),
                Paragraph(status_change or "–", normal_style),
            ])
        hist_table = Table(hist_data, colWidths=["22%", "24%", "22%", "32%"])
        hist_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E40AF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(hist_table)
    else:
        story.append(Paragraph("No history available.", normal_style))

    # ==========================================
    # FOOTER
    # ==========================================
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"This report was automatically generated by SmartClaim Portal on {now}. "
        "This document is confidential and intended solely for the addressee. "
        "For any queries, please contact support@smartclaim.com.",
        footer_style,
    ))

    doc.build(story)
    return buf.getvalue()
