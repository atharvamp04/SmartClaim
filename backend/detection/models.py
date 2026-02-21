from django.db import models
from django.contrib.postgres.fields import JSONField  # Use this if using PostgreSQL
from django.contrib.auth.models import User
# OR for Django 3.1+
# from django.db.models import JSONField

# ============================================================
# NEW: User Profile for Role Management
# ============================================================

class UserProfile(models.Model):
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('customer', 'Customer'),
        ('surveyor', 'Surveyor'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='customer')
    employee_id = models.CharField(max_length=50, blank=True, null=True)
    assigned_region = models.CharField(max_length=100, blank=True, null=True)
    phone_number = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self):
        return f"{self.user.username} ({self.role})"

    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'

class Policyholder(models.Model):
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=100, unique=True)
    sex = models.CharField(max_length=10)
    marital_status = models.CharField(max_length=20, blank=True, null=True)
    age = models.PositiveIntegerField()
    address_area = models.CharField(max_length=50, blank=True, null=True)
    policy_type = models.CharField(max_length=50)
    base_policy = models.CharField(max_length=50)
    number_of_cars = models.PositiveIntegerField(default=1)
    agent_type = models.CharField(max_length=50, blank=True, null=True)
    vehicle_make = models.CharField(max_length=50)
    vehicle_category = models.CharField(max_length=50)
    vehicle_price_category = models.CharField(max_length=50, blank=True, null=True)
    age_of_vehicle = models.CharField(max_length=50, blank=True, null=True)
    year_of_vehicle = models.PositiveIntegerField(blank=True, null=True)
    driver_rating = models.PositiveIntegerField(blank=True, null=True)
    past_number_of_claims = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    vehicle_model = models.CharField(max_length=100, default='default')

    def __str__(self):
        return self.username

# ============================================================
# EXISTING + MODIFIED: Claim (surveyor fields added)
# ============================================================

class Claim(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending Review'),
        ('Under Survey', 'Under Survey'),       # NEW
        ('Survey Completed', 'Survey Completed'), # NEW
        ('Verified', 'Verified/Claimed'),
        ('Fraud', 'Fraud Detected'),
        ('Rejected', 'Rejected'),
    ]

    RISK_LEVEL_CHOICES = [
        ('LOW', 'Low Risk'),
        ('MEDIUM', 'Medium Risk'),
        ('HIGH', 'High Risk'),
        ('CRITICAL', 'Critical Risk'),
    ]

    SEVERITY_CHOICES = [
        ('LOW', 'Low Severity'),
        ('MEDIUM', 'Medium Severity'),
        ('HIGH', 'High Severity'),
    ]

    SURVEYOR_RECOMMENDATION_CHOICES = [        # NEW
        ('APPROVE', 'Approve Claim'),
        ('REJECT', 'Reject Claim'),
        ('INVESTIGATE', 'Further Investigation'),
        ('PARTIAL', 'Partial Approval'),
    ]

    # --- Existing fields (unchanged) ---
    policyholder = models.ForeignKey(
        Policyholder,
        on_delete=models.CASCADE,
        related_name='claims'
    )
    claim_number = models.CharField(max_length=50, unique=True, editable=False)
    claim_description = models.TextField()
    accident_date = models.DateField()
    claim_amount = models.DecimalField(max_digits=10, decimal_places=2)
    dl_number = models.CharField(max_length=50, verbose_name="Driving License Number")
    vehicle_reg_no = models.CharField(max_length=50, verbose_name="Vehicle Registration Number")
    fir_number = models.CharField(max_length=50, blank=True, null=True, verbose_name="FIR Number")
    total_images_submitted = models.PositiveIntegerField(default=1)
    image_paths = models.JSONField(default=list, blank=True)
    fraud_detected = models.BooleanField(default=False)
    confidence_score = models.DecimalField(max_digits=5, decimal_places=2)
    risk_level = models.CharField(max_length=10, choices=RISK_LEVEL_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    tabular_fraud_probability = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    image_fraud_probability = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    fusion_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    overall_damage_severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, null=True, blank=True)
    total_damage_areas = models.PositiveIntegerField(default=0)
    average_damage_percentage = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    max_damage_percentage = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    max_fraud_image_index = models.PositiveIntegerField(null=True, blank=True)
    max_fraud_probability = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    fraud_probability_mean = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    fraud_probability_std = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    dl_verification_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    rto_verification_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    fir_verification_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    verification_reliability = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    detailed_analysis = models.JSONField(default=dict, blank=True)
    annotated_images = models.JSONField(default=list, blank=True)
    admin_notes = models.TextField(blank=True, null=True)
    rejection_reason = models.TextField(
        blank=True,
        null=True,
        verbose_name="Rejection Reason",
        help_text="Required when claim is rejected – will be visible to the customer."
    )
    reviewed_by = models.CharField(max_length=100, blank=True, null=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    submitted_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # --- NEW: Surveyor Assignment Fields ---
    assigned_surveyor = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='assigned_claims',
        verbose_name="Assigned Surveyor"
    )
    assigned_at = models.DateTimeField(null=True, blank=True)

    # --- NEW: Surveyor Report Fields ---
    surveyor_notes = models.TextField(
        blank=True,
        null=True,
        verbose_name="Field Survey Notes"
    )
    surveyor_recommendation = models.CharField(
        max_length=20,
        choices=SURVEYOR_RECOMMENDATION_CHOICES,
        blank=True,
        null=True,
        verbose_name="Surveyor Recommendation"
    )
    surveyor_assessed_amount = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        verbose_name="Surveyor Assessed Amount"
    )
    damage_verified = models.BooleanField(
        null=True,
        blank=True,
        verbose_name="Damage Verified on Site"
    )
    survey_completed_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="Survey Completion Time"
    )

    class Meta:
        ordering = ['-submitted_at']
        verbose_name = 'Claim'
        verbose_name_plural = 'Claims'
        indexes = [
            models.Index(fields=['status', '-submitted_at']),
            models.Index(fields=['policyholder', '-submitted_at']),
            models.Index(fields=['fraud_detected', 'status']),
            models.Index(fields=['assigned_surveyor', 'status']),  # NEW index
        ]

    def __str__(self):
        return f"{self.claim_number} - {self.policyholder.username} - {self.status}"

    def save(self, *args, **kwargs):
        if not self.claim_number:
            from django.utils import timezone
            timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
            self.claim_number = f"CLM-{timestamp}-{self.policyholder.id}"

        if self.fraud_detected and self.status == 'Pending':
            self.status = 'Fraud'

        super().save(*args, **kwargs)

    @property
    def is_high_risk(self):
        return self.risk_level in ['HIGH', 'CRITICAL'] or self.fraud_detected

    @property
    def requires_review(self):
        return self.status in ['Pending', 'Survey Completed'] or self.risk_level in ['MEDIUM', 'HIGH', 'CRITICAL']

    @property
    def is_assigned_to_surveyor(self):           # NEW
        return self.assigned_surveyor is not None

    @property
    def final_claim_amount(self):                # NEW
        """Returns surveyor amount if available, else AI-calculated amount"""
        return self.surveyor_assessed_amount or self.claim_amount

class ClaimImage(models.Model):
    """Separate model for individual claim images with detailed analysis"""
    claim = models.ForeignKey(Claim, on_delete=models.CASCADE, related_name='images')
    image_index = models.PositiveIntegerField()
    image_file = models.ImageField(upload_to='claim_images/%Y/%m/%d/')
    
    # Individual Image Analysis
    fraud_probability = models.DecimalField(max_digits=5, decimal_places=2)
    confidence = models.DecimalField(max_digits=5, decimal_places=2)
    damage_percentage = models.DecimalField(max_digits=5, decimal_places=2)
    severity_level = models.CharField(max_length=10, choices=Claim.SEVERITY_CHOICES)
    damage_areas_count = models.PositiveIntegerField(default=0)
    
    # Annotated Image
    annotated_image = models.ImageField(
        upload_to='annotated_images/%Y/%m/%d/', 
        null=True, 
        blank=True
    )
    
    # Detailed damage regions data
    damage_regions = models.JSONField(default=list, blank=True)
    
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['image_index']
        unique_together = ['claim', 'image_index']
    
    def __str__(self):
        return f"Image {self.image_index} - {self.claim.claim_number}"
    
class SurveyorFieldPhoto(models.Model):
    """Photos uploaded by surveyor during field visit"""
    claim = models.ForeignKey(Claim, on_delete=models.CASCADE, related_name='field_photos')
    photo = models.ImageField(upload_to='surveyor_photos/%Y/%m/%d/')
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    caption = models.CharField(max_length=255, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Field Photo - {self.claim.claim_number} - {self.uploaded_at}"

    class Meta:
        ordering = ['uploaded_at']
        verbose_name = 'Surveyor Field Photo'
        verbose_name_plural = 'Surveyor Field Photos'


# ============================================================
# EXISTING + MODIFIED: ClaimHistory (new action choices added)
# ============================================================

class ClaimHistory(models.Model):
    ACTION_CHOICES = [
        ('submitted', 'Claim Submitted'),
        ('analyzed', 'AI Analysis Completed'),
        ('reviewed', 'Admin Reviewed'),
        ('status_changed', 'Status Changed'),
        ('approved', 'Claim Approved'),
        ('rejected', 'Claim Rejected'),
        ('surveyor_assigned', 'Surveyor Assigned'),       # NEW
        ('survey_started', 'Field Survey Started'),        # NEW
        ('survey_completed', 'Field Survey Completed'),    # NEW
        ('surveyor_report', 'Surveyor Report Submitted'),  # NEW
    ]

    claim = models.ForeignKey(Claim, on_delete=models.CASCADE, related_name='history')
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    old_status = models.CharField(max_length=20, blank=True, null=True)
    new_status = models.CharField(max_length=20, blank=True, null=True)
    performed_by = models.CharField(max_length=100)
    notes = models.TextField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Claim History'
        verbose_name_plural = 'Claim Histories'

    def __str__(self):
        return f"{self.claim.claim_number} - {self.action} - {self.timestamp}"

# IMPORTANT: Admin class must NOT be indented - it should be at the left margin
class Admin(models.Model):
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=255)  # Hashed password
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.username} ({self.email})"
    
    class Meta:
        verbose_name = 'Admin User'
        verbose_name_plural = 'Admin Users'