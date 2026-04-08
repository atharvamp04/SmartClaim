from django.contrib import admin
from django.urls import path, include
from django.conf import settings  # ← ADD THIS LINE
from django.conf.urls.static import static
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from detection.views import login_with_role

urlpatterns = [
    path('admin/', admin.site.urls),

    # Detection app (trailing slash and without — proxy strips slashes)
    path('api/detection/', include('detection.urls')),

    # Auth routes — with and without trailing slash
    path('api/auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/auth/login', TokenObtainPairView.as_view(), name='token_obtain_pair_noslash'),
    path('api/auth/login-with-role/', login_with_role, name='login_with_role'),
    path('api/auth/login-with-role', login_with_role, name='login_with_role_noslash'),
    path('api/auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/auth/refresh', TokenRefreshView.as_view(), name='token_refresh_noslash'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)