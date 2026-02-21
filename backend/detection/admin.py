from django.contrib import admin
from .models import UserProfile, Policyholder, Claim, ClaimHistory, SurveyorFieldPhoto

# Register UserProfile with admin customization
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'employee_id', 'assigned_region', 'phone_number')
    list_filter = ('role',)
    search_fields = ('user__username', 'user__email', 'employee_id')
    list_editable = ('role', 'assigned_region')
    
    fieldsets = (
        ('User Info', {
            'fields': ('user',)
        }),
        ('Role Assignment', {
            'fields': ('role',)
        }),
        ('Additional Info', {
            'fields': ('employee_id', 'assigned_region', 'phone_number')
        }),
    )


@admin.register(Policyholder)
class PolicyholderAdmin(admin.ModelAdmin):
    list_display = ('username', 'email', 'vehicle_make', 'policy_type', 'created_at')
    search_fields = ('username', 'email')
    list_filter = ('policy_type', 'vehicle_category', 'created_at')


@admin.register(Claim)
class ClaimAdmin(admin.ModelAdmin):
    list_display = ('claim_number', 'policyholder', 'status', 'risk_level', 'claim_amount', 'submitted_at')
    list_filter = ('status', 'risk_level', 'submitted_at')
    search_fields = ('claim_number', 'policyholder__username')
    readonly_fields = ('claim_number', 'submitted_at', 'updated_at')


@admin.register(ClaimHistory)
class ClaimHistoryAdmin(admin.ModelAdmin):
    list_display = ('claim', 'action', 'timestamp', 'performed_by')
    list_filter = ('action', 'timestamp')
    search_fields = ('claim__claim_number',)


@admin.register(SurveyorFieldPhoto)
class SurveyorFieldPhotoAdmin(admin.ModelAdmin):
    list_display = ('claim', 'uploaded_by', 'uploaded_at')
    list_filter = ('uploaded_at',)
    search_fields = ('claim__claim_number',)

