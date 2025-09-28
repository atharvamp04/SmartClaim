# detection/urls.py
from django.urls import path
from .views import (
    # ML Prediction Views

    predict_claim,  # Add this import
    
    # Authentication & User Views
    RegisterView,
    protected_view,
    
    # Policyholder Views
    PolicyholderCreateView,
    policyholder_detail,
    
    # Health Check
    health_check
)

urlpatterns = [
    # ML Prediction endpoints
    path('predict-claim/', predict_claim, name='predict_claim'),  # Add this line - Predict using username + image
    
    # Authentication endpoints
    path('register/', RegisterView.as_view(), name='register'),
    path('protected/', protected_view, name='protected'),
    
    # Policyholder endpoints
    path('policyholder/create/', PolicyholderCreateView.as_view(), name='policyholder-create'),
    path('policyholder/<str:username>/', policyholder_detail, name='policyholder-detail'),
    
    # System health check
    path('health/', health_check, name='health_check'),
]