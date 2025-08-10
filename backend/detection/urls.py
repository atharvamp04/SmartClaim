from django.urls import path, include
from .views import protected_view, RegisterView
from .views import PolicyholderCreateView, policyholder_detail

urlpatterns = [
    path('protected/', protected_view, name='protected'),
    path('register/', RegisterView.as_view(), name='register'),
    path('policyholders/', PolicyholderCreateView.as_view(), name='policyholder-create'),
    path('policyholders/<str:username>/', policyholder_detail, name='policyholder-detail'),
]
