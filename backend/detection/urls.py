# detection/urls.py
from django.urls import path
from .views import (
    # ML Prediction Views
    predict_claim,  # ✅ Main prediction endpoint
    
    # Authentication & User Views
    RegisterView,
    protected_view,
    
    # Policyholder Views
    PolicyholderCreateView,
    policyholder_detail,
    
    # Health Check
    health_check
)

# Import claim management views
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
    claim_api_info
)

urlpatterns = [
    # ==========================================
    # ML PREDICTION ENDPOINTS
    # ==========================================
    path('predict-claim/', predict_claim, name='predict_claim'),  # Main fraud detection endpoint
    
    # ==========================================
    # AUTHENTICATION ENDPOINTS
    # ==========================================
    path('register/', RegisterView.as_view(), name='register'),
    path('protected/', protected_view, name='protected'),
    
    # ==========================================
    # POLICYHOLDER ENDPOINTS
    # ==========================================
    path('policyholder/create/', PolicyholderCreateView.as_view(), name='policyholder-create'),
    path('policyholder/<str:username>/', policyholder_detail, name='policyholder-detail'),
    
    # ==========================================
    # CLAIM RETRIEVAL ENDPOINTS
    # ==========================================
    path('claims/', get_all_claims, name='get-all-claims'),  # GET all claims with filters
    path('claims/<int:claim_id>/', get_claim_detail, name='get-claim-detail'),  # GET claim by ID
    path('claims/number/<str:claim_number>/', get_claim_by_number, name='get-claim-by-number'),  # GET by claim number
    path('claims/pending/', get_pending_claims, name='get-pending-claims'),  # GET pending claims
    path('claims/verified/', get_verified_claims, name='get-verified-claims'),  # GET verified claims
    path('claims/fraud/', get_fraud_claims, name='get-fraud-claims'),  # GET fraud-detected claims
    path('claims/high-risk/', get_high_risk_claims, name='get-high-risk-claims'),  # GET high-risk claims
    
    # ==========================================
    # CLAIM MANAGEMENT ENDPOINTS
    # ==========================================
    path('claims/<int:claim_id>/status/', update_claim_status, name='update-claim-status'),  # POST to update status
    path('claims/<int:claim_id>/review/', add_admin_review, name='add-admin-review'),  # POST to add review
    path('claims/<int:claim_id>/delete/', delete_claim, name='delete-claim'),  # DELETE claim
    
    # ==========================================
    # CLAIM HISTORY & STATISTICS
    # ==========================================
    path('claims/<int:claim_id>/history/', get_claim_history, name='get-claim-history'),  # GET claim history
    path('claims/statistics/', get_claim_statistics, name='get-claim-statistics'),  # GET overall statistics
    path('claims/policyholder/<str:username>/', get_policyholder_claims, name='get-policyholder-claims'),  # GET claims by policyholder
    
    # ==========================================
    # SEARCH & API INFO
    # ==========================================
    path('claims/search/', search_claims, name='search-claims'),  # GET search claims
    path('claims/info/', claim_api_info, name='claim-api-info'),  # GET API documentation
    
    # ==========================================
    # SYSTEM HEALTH CHECK
    # ==========================================
    path('health/', health_check, name='health_check'),  # System health and model status
]