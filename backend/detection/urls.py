from django.urls import path
from .views import protected_view, RegisterView

urlpatterns = [
    path('protected/', protected_view, name='protected'),
    path('register/', RegisterView.as_view(), name='register'),
]
