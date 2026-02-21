# detection/report_views.py
"""
Views for:
  1. Admin approve/reject with rejection reason + PDF email
  2. Customer portal: fetching claim notifications
  3. PDF download endpoint (admin + customer)
"""
import logging
from django.http import HttpResponse
from django.core.mail import EmailMessage
from django.conf import settings
from django.utils import timezone

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status

from .models import Claim, ClaimHistory, Policyholder
from .pdf_report import generate_claim_report_pdf, REPORTLAB_AVAILABLE
from .claim_handler import ClaimDatabaseHandler

logger = logging.getLogger(__name__)


# ============================================================
# HELPER: Send email with PDF attached
# ============================================================

def _is_email_configured() -> bool:
    """Return True only if SMTP credentials are set in settings."""
    host_user = getattr(settings, "EMAIL_HOST_USER", "")
    host_pass = getattr(settings, "EMAIL_HOST_PASSWORD", "")
    return bool(host_user and host_pass)


def _reportlab_available() -> bool:
    """Re-check at call-time (avoids stale import-time flag after install)."""
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


def _send_decision_email(claim, pdf_bytes: bytes):
    """
    Send claim decision email to policyholder with PDF attached.
    Returns (sent: bool, reason: str).
    """
    # ── Guard: no SMTP credentials configured ──────────────────────
    if not _is_email_configured():
        msg = (
            "Email not configured. Set EMAIL_HOST_USER and EMAIL_HOST_PASSWORD "
            "environment variables to enable email notifications."
        )
        logger.info(msg)
        return False, msg

    try:
        ph = claim.policyholder
        if not ph or not ph.email:
            msg = f"No email address on file for policyholder of claim {claim.claim_number}"
            logger.warning(msg)
            return False, msg

        is_approved = claim.status == "Verified"
        subject_map = {
            "Verified": f"Your Claim {claim.claim_number} Has Been Approved - SmartClaim",
            "Rejected": f"Your Claim {claim.claim_number} Has Been Rejected - SmartClaim",
        }
        subject = subject_map.get(claim.status, f"Update on Claim {claim.claim_number} - SmartClaim")

        portal_url = getattr(settings, "SMARTCLAIM_PORTAL_URL", "http://localhost:3000")

        if is_approved:
            body = (
                f"Dear {ph.username},\n\n"
                f"We are pleased to inform you that your insurance claim has been APPROVED.\n\n"
                f"Claim Number : {claim.claim_number}\n"
                f"Status       : Approved\n"
                f"Final Amount : Rs.{float(claim.surveyor_assessed_amount or claim.claim_amount):,.2f}\n"
                f"Reviewed By  : {claim.reviewed_by or 'Admin'}\n"
                f"Date         : {claim.reviewed_at.strftime('%d %B %Y') if claim.reviewed_at else 'N/A'}\n\n"
                f"Please find the complete claim report attached as a PDF.\n\n"
                f"Log in to view your report: {portal_url}\n\n"
                f"Best regards,\nSmartClaim Team"
            )
        else:
            body = (
                f"Dear {ph.username},\n\n"
                f"We regret to inform you that your insurance claim has been REJECTED.\n\n"
                f"Claim Number : {claim.claim_number}\n"
                f"Status       : Rejected\n"
                f"Reviewed By  : {claim.reviewed_by or 'Admin'}\n"
                f"Date         : {claim.reviewed_at.strftime('%d %B %Y') if claim.reviewed_at else 'N/A'}\n\n"
                f"Reason for Rejection:\n"
                f"{claim.rejection_reason or 'Please contact support for more details.'}\n\n"
                f"If you believe this decision is incorrect, please contact our support team.\n"
                f"The complete claim report is attached as a PDF for your records.\n\n"
                f"Log in to view: {portal_url}\n\n"
                f"Best regards,\nSmartClaim Team"
            )

        email = EmailMessage(
            subject=subject,
            body=body,
            from_email=settings.EMAIL_HOST_USER,   # must match authenticated sender
            to=[ph.email],
        )

        if pdf_bytes:
            email.attach(
                filename=f"SmartClaim_Report_{claim.claim_number}.pdf",
                content=pdf_bytes,
                mimetype="application/pdf",
            )

        email.send(fail_silently=False)
        logger.info(f"Decision email sent to {ph.email} for claim {claim.claim_number}")
        return True, "Email sent successfully"

    except Exception as e:
        err_msg = str(e)
        logger.error(f"Failed to send email for claim {claim.claim_number}: {err_msg}")
        return False, err_msg


# ============================================================
# ENDPOINT 1: Admin Final Decision (Approve / Reject)
# POST /api/detection/claims/<claim_id>/decision/
# Body: { status: "Verified"|"Rejected", rejection_reason?: "...", admin_notes?: "..." }
# ============================================================

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def admin_final_decision(request, claim_id):
    """
    Admin approves or rejects a claim (typically after survey is complete).

    Body:
        status          – "Verified" or "Rejected"
        rejection_reason– (required when status == "Rejected") visible to customer
        admin_notes     – optional internal notes
        send_email      – default True; set False to skip email
    """
    try:
        new_status = request.data.get("status")
        rejection_reason = request.data.get("rejection_reason", "").strip()
        admin_notes = request.data.get("admin_notes", "").strip()
        send_email = request.data.get("send_email", True)
        admin_user = request.user.username

        # --- Validations ---
        if new_status not in ("Verified", "Rejected"):
            return Response(
                {"error": "status must be 'Verified' or 'Rejected'"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if new_status == "Rejected" and not rejection_reason:
            return Response(
                {"error": "rejection_reason is required when rejecting a claim."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            claim = Claim.objects.select_related("policyholder", "assigned_surveyor").get(pk=claim_id)
        except Claim.DoesNotExist:
            return Response({"error": "Claim not found"}, status=status.HTTP_404_NOT_FOUND)

        old_status = claim.status

        # --- Update claim ---
        claim.status = new_status
        if admin_notes:
            claim.admin_notes = admin_notes
        if new_status == "Rejected":
            claim.rejection_reason = rejection_reason
        else:
            claim.rejection_reason = None  # clear if previously rejected

        claim.reviewed_by = admin_user
        claim.reviewed_at = timezone.now()
        claim.save()

        # --- Log history ---
        action_map = {"Verified": "approved", "Rejected": "rejected"}
        ClaimHistory.objects.create(
            claim=claim,
            action=action_map.get(new_status, "status_changed"),
            old_status=old_status,
            new_status=new_status,
            performed_by=admin_user,
            notes=(
                f"Admin decision: {new_status}. "
                + (f"Reason: {rejection_reason}" if rejection_reason else "")
                + (f" Notes: {admin_notes}" if admin_notes else "")
            ).strip(),
        )

        # --- Generate PDF ---
        pdf_bytes = None
        pdf_generated = False
        pdf_error = None
        try:
            pdf_bytes = generate_claim_report_pdf(claim)
            pdf_generated = True
        except Exception as e:
            pdf_error = str(e)
            logger.error(f"PDF generation failed for claim {claim_id}: {e}")

        # --- Send Email ---
        email_sent = False
        email_error = None
        if send_email:
            email_sent, email_error = _send_decision_email(claim, pdf_bytes)

        claim_data = ClaimDatabaseHandler.serialize_claim(
            claim, include_images=False, include_history=True
        )

        return Response({
            "message": f"Claim {new_status.lower()} successfully.",
            "claim": claim_data,
            "pdf_generated": pdf_generated,
            "pdf_error": pdf_error,
            "email_sent": email_sent,
            "email_error": email_error,
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.exception(f"admin_final_decision error: {e}")
        return Response(
            {"error": f"Internal server error: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ============================================================
# ENDPOINT 2: Download PDF for a claim
# GET /api/detection/claims/<claim_id>/report-pdf/
# Accessible by both admin and the claim's policyholder
# ============================================================

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def download_claim_pdf(request, claim_id):
    """Download the PDF report for a specific claim."""
    try:
        if not REPORTLAB_AVAILABLE:
            return Response(
                {"error": "PDF generation library (reportlab) is not installed on the server."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            claim = Claim.objects.select_related(
                "policyholder", "assigned_surveyor"
            ).prefetch_related("history").get(pk=claim_id)
        except Claim.DoesNotExist:
            return Response({"error": "Claim not found"}, status=status.HTTP_404_NOT_FOUND)

        # Re-check reportlab at call-time (handles case where it was installed
        # after the server started and the module-level flag is stale)
        if not _reportlab_available():
            return Response(
                {"error": "PDF generation library (reportlab) is not installed. Run: pip install reportlab"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # Authorization: admin or the policyholder whose claim this is
        user = request.user
        is_admin = user.is_staff or (hasattr(user, "profile") and user.profile.role == "admin")
        is_owner = claim.policyholder and claim.policyholder.username == user.username

        if not is_admin and not is_owner:
            return Response(
                {"error": "You do not have permission to download this report."},
                status=status.HTTP_403_FORBIDDEN,
            )

        pdf_bytes = generate_claim_report_pdf(claim)

        resp = HttpResponse(pdf_bytes, content_type="application/pdf")
        resp["Content-Disposition"] = (
            f'attachment; filename="SmartClaim_Report_{claim.claim_number}.pdf"'
        )
        return resp

    except Exception as e:
        logger.exception(f"download_claim_pdf error: {e}")
        return Response(
            {"error": f"PDF generation failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ============================================================
# ENDPOINT 3: Customer Notifications
# GET /api/detection/customer/notifications/
# Returns recent finalized claims for the logged-in customer
# ============================================================

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def customer_notifications(request):
    """
    Returns notifications for the currently logged-in customer.

    A notification is generated for every claim that has a final decision
    (Verified, Rejected) or important status change.
    """
    try:
        username = request.user.username

        try:
            ph = Policyholder.objects.get(username=username)
        except Policyholder.DoesNotExist:
            # User may not have filled in their policyholder profile yet
            return Response({"notifications": [], "unread_count": 0})

        # All claims for this policyholder, ordered by most recent activity
        claims = Claim.objects.filter(policyholder=ph).order_by("-updated_at")

        notifications = []
        for claim in claims:
            notif_type = None
            title = None
            message = None
            action_available = claim.status in ("Verified", "Rejected")

            if claim.status == "Verified":
                notif_type = "success"
                title = "Claim Approved ✅"
                message = (
                    f"Your claim {claim.claim_number} has been approved. "
                    f"Amount: ₹{float(claim.surveyor_assessed_amount or claim.claim_amount):,.0f}"
                )
            elif claim.status == "Rejected":
                notif_type = "error"
                title = "Claim Rejected ❌"
                message = (
                    f"Your claim {claim.claim_number} has been rejected. "
                    + (f"Reason: {claim.rejection_reason}" if claim.rejection_reason else "Please contact support for details.")
                )
            elif claim.status == "Fraud":
                notif_type = "warning"
                title = "Fraud Alert ⚠️"
                message = f"Claim {claim.claim_number} has been flagged for potential fraud."
            elif claim.status == "Under Survey":
                notif_type = "info"
                title = "Survey In Progress 🔍"
                message = f"Claim {claim.claim_number} is being surveyed by a field agent."
            elif claim.status == "Survey Completed":
                notif_type = "info"
                title = "Survey Completed 📋"
                message = f"Field survey for claim {claim.claim_number} is complete. Awaiting admin decision."
            elif claim.status == "Pending":
                notif_type = "pending"
                title = "Claim Under Review ⏳"
                message = f"Claim {claim.claim_number} is being reviewed by our team."

            if notif_type:
                notifications.append({
                    "claim_id": claim.id,
                    "claim_number": claim.claim_number,
                    "type": notif_type,
                    "title": title,
                    "message": message,
                    "status": claim.status,
                    "updated_at": claim.updated_at.isoformat(),
                    "action_available": action_available,
                    "rejection_reason": claim.rejection_reason if claim.status == "Rejected" else None,
                    "claim_amount": float(claim.surveyor_assessed_amount or claim.claim_amount),
                })

        unread_count = sum(1 for n in notifications if n["status"] in ("Verified", "Rejected", "Fraud"))

        return Response({
            "notifications": notifications,
            "unread_count": unread_count,
            "total": len(notifications),
        })

    except Exception as e:
        logger.exception(f"customer_notifications error: {e}")
        return Response(
            {"error": f"Failed to fetch notifications: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ============================================================
# ENDPOINT 4: Get single claim details for customer (with rejection)
# GET /api/detection/customer/claims/
# ============================================================

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def customer_my_claims(request):
    """Return all claims belonging to the currently logged-in customer."""
    try:
        username = request.user.username

        try:
            ph = Policyholder.objects.get(username=username)
        except Policyholder.DoesNotExist:
            return Response({"claims": [], "total": 0})

        claims = Claim.objects.filter(policyholder=ph).order_by("-submitted_at")

        claims_data = []
        for claim in claims:
            claims_data.append({
                "id": claim.id,
                "claim_number": claim.claim_number,
                "status": claim.status,
                "claim_amount": float(claim.surveyor_assessed_amount or claim.claim_amount),
                "risk_level": claim.risk_level,
                "fraud_detected": claim.fraud_detected,
                "accident_date": claim.accident_date.isoformat() if claim.accident_date else None,
                "submitted_at": claim.submitted_at.isoformat() if claim.submitted_at else None,
                "updated_at": claim.updated_at.isoformat() if claim.updated_at else None,
                "reviewed_by": claim.reviewed_by,
                "reviewed_at": claim.reviewed_at.isoformat() if claim.reviewed_at else None,
                "admin_notes": claim.admin_notes,
                "rejection_reason": claim.rejection_reason,
                "has_report": claim.status in ("Verified", "Rejected"),
            })

        return Response({
            "claims": claims_data,
            "total": len(claims_data),
        })

    except Exception as e:
        logger.exception(f"customer_my_claims error: {e}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
