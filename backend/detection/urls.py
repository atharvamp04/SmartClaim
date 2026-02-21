# detection/urls.py
from django.urls import path
from .views import (
    # ML Prediction Views
    predict_claim,

    # Authentication & User Views
    RegisterView,
    protected_view,
    login_with_role,          # NEW

    # Policyholder Views
    PolicyholderCreateView,
    policyholder_detail,

    # Surveyor Views                  NEW
    surveyor_assigned_claims,
    surveyor_submit_report,

    # Admin Surveyor Management       NEW
    admin_assign_surveyor,
    list_surveyors,
    create_surveyor,

    # Health Check
    health_check,
)

from .claim_views import (
    # Claim Retrieval
    get_all_claims,
    get_claim_detail,
    get_claim_by_number,
    get_pending_claims,
    get_verified_claims,
    get_fraud_claims,
    get_high_risk_claims,

    # Claim Management
    update_claim_status,
    add_admin_review,
    delete_claim,

    # Claim History & Statistics
    get_claim_history,
    get_claim_statistics,
    get_policyholder_claims,

    # Search & Info
    search_claims,
    claim_api_info,

    # NEW: Surveyor claim views
    get_under_survey_claims,
    get_survey_completed_claims,
)

from .report_views import (
    admin_final_decision,
    download_claim_pdf,
    customer_notifications,
    customer_my_claims,
)

urlpatterns = [

    # ==========================================
    # ML PREDICTION ENDPOINTS
    # ==========================================
    path('predict-claim/', predict_claim, name='predict_claim'),

    # ==========================================
    # AUTHENTICATION ENDPOINTS
    # ==========================================
    path('register/', RegisterView.as_view(), name='register'),
    path('protected/', protected_view, name='protected'),
    path('login-with-role/', login_with_role, name='login_with_role'),  # NEW

    # ==========================================
    # POLICYHOLDER ENDPOINTS
    # ==========================================
    path('policyholder/create/', PolicyholderCreateView.as_view(), name='policyholder-create'),
    path('policyholder/<str:username>/', policyholder_detail, name='policyholder-detail'),

    # ==========================================
    # CLAIM RETRIEVAL ENDPOINTS
    # ==========================================
    path('claims/', get_all_claims, name='get-all-claims'),
    path('claims/<int:claim_id>/', get_claim_detail, name='get-claim-detail'),
    path('claims/number/<str:claim_number>/', get_claim_by_number, name='get-claim-by-number'),
    path('claims/pending/', get_pending_claims, name='get-pending-claims'),
    path('claims/verified/', get_verified_claims, name='get-verified-claims'),
    path('claims/fraud/', get_fraud_claims, name='get-fraud-claims'),
    path('claims/high-risk/', get_high_risk_claims, name='get-high-risk-claims'),
    path('claims/under-survey/', get_under_survey_claims, name='get-under-survey-claims'),          # NEW
    path('claims/survey-completed/', get_survey_completed_claims, name='get-survey-completed-claims'),  # NEW

    # ==========================================
    # CLAIM MANAGEMENT ENDPOINTS
    # ==========================================
    path('claims/<int:claim_id>/status/', update_claim_status, name='update-claim-status'),
    path('claims/<int:claim_id>/review/', add_admin_review, name='add-admin-review'),
    path('claims/<int:claim_id>/delete/', delete_claim, name='delete-claim'),

    # NEW: Admin assigns surveyor to a specific claim
    path('claims/<int:claim_id>/assign-surveyor/', admin_assign_surveyor, name='assign-surveyor'),

    # ==========================================
    # CLAIM HISTORY & STATISTICS
    # ==========================================
    path('claims/<int:claim_id>/history/', get_claim_history, name='get-claim-history'),
    path('claims/statistics/', get_claim_statistics, name='get-claim-statistics'),
    path('claims/policyholder/<str:username>/', get_policyholder_claims, name='get-policyholder-claims'),

    # ==========================================
    # SEARCH & API INFO
    # ==========================================
    path('claims/search/', search_claims, name='search-claims'),
    path('claims/info/', claim_api_info, name='claim-api-info'),

    # ==========================================
    # SURVEYOR ENDPOINTS                        NEW
    # ==========================================

    # Surveyor dashboard — see their assigned claims
    path('surveyor/claims/', surveyor_assigned_claims, name='surveyor-claims'),

    # Surveyor submits field report for a specific claim
    path('surveyor/claims/<int:claim_id>/report/', surveyor_submit_report, name='surveyor-report'),

    # ==========================================
    # ADMIN — SURVEYOR MANAGEMENT              NEW
    # ==========================================

    # List all surveyors (for admin assign-surveyor dropdown)
    path('admin/surveyors/', list_surveyors, name='list-surveyors'),

    # Create a new surveyor account
    path('admin/surveyors/create/', create_surveyor, name='create-surveyor'),

    # ==========================================
    # ADMIN FINAL DECISION + PDF REPORT         NEW
    # ==========================================

    # Admin approve or reject claim (with rejection reason + auto-email)
    path('claims/<int:claim_id>/decision/', admin_final_decision, name='admin-final-decision'),

    # Download PDF report for a claim (admin or policyholder)
    path('claims/<int:claim_id>/report-pdf/', download_claim_pdf, name='download-claim-pdf'),

    # ==========================================
    # CUSTOMER PORTAL                           NEW
    # ==========================================

    # Get all claims for logged-in customer (with status, rejection reason, etc.)
    path('customer/claims/', customer_my_claims, name='customer-my-claims'),

    # Get notifications for logged-in customer
    path('customer/notifications/', customer_notifications, name='customer-notifications'),

    # ==========================================
    # SYSTEM HEALTH CHECK
    # ==========================================
    path('health/', health_check, name='health_check'),
]