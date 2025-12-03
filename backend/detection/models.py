from django.db import models
from django.contrib.postgres.fields import JSONField  # Use this if using PostgreSQL
# OR for Django 3.1+
# from django.db.models import JSONField

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

    def __str__(self):
        return self.username


class Claim(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending Review'),
        ('Verified', 'Verified/Claimed'),
        ('Fraud', 'Fraud Detected'),
        ('Rejected', 'Rejected'),
    ]
    
    RISK_LEVEL_CHOICES = [
        ('LOW', 'Low Risk'),
        ('MEDIUM', 'Medium Risk'),
        ('HIGH', 'High Risk'),
    ]
    
    SEVERITY_CHOICES = [
        ('LOW', 'Low Severity'),
        ('MEDIUM', 'Medium Severity'),
        ('HIGH', 'High Severity'),
    ]

    # Foreign Key to Policyholder
    policyholder = models.ForeignKey(
        Policyholder, 
        on_delete=models.CASCADE, 
        related_name='claims'
    )
    
    # Basic Claim Information
    claim_number = models.CharField(max_length=50, unique=True, editable=False)
    claim_description = models.TextField()
    accident_date = models.DateField()
    claim_amount = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Verification Documents
    dl_number = models.CharField(max_length=50, verbose_name="Driving License Number")
    vehicle_reg_no = models.CharField(max_length=50, verbose_name="Vehicle Registration Number")
    fir_number = models.CharField(max_length=50, blank=True, null=True, verbose_name="FIR Number")
    
    # Images
    total_images_submitted = models.PositiveIntegerField(default=1)
    # Store image paths or URLs in a JSON field
    image_paths = models.JSONField(default=list, blank=True)
    
    # Fraud Detection Results
    fraud_detected = models.BooleanField(default=False)
    confidence_score = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        help_text="Confidence score as percentage (0-100)"
    )
    risk_level = models.CharField(max_length=10, choices=RISK_LEVEL_CHOICES)
    
    # Status and Processing
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    
    # Analysis Scores (stored as percentages 0-100)
    tabular_fraud_probability = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        null=True, 
        blank=True,
        help_text="Tabular model fraud probability %"
    )
    image_fraud_probability = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        null=True, 
        blank=True,
        help_text="Image model fraud probability %"
    )
    fusion_score = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        null=True, 
        blank=True,
        help_text="Final fusion score %"
    )
    
    # Damage Detection Results
    overall_damage_severity = models.CharField(
        max_length=10, 
        choices=SEVERITY_CHOICES, 
        null=True, 
        blank=True
    )
    total_damage_areas = models.PositiveIntegerField(default=0)
    average_damage_percentage = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        null=True, 
        blank=True,
        help_text="Average damage coverage across all images"
    )
    
    # Multi-Image Analysis Metrics
    max_fraud_image_index = models.PositiveIntegerField(null=True, blank=True)
    max_fraud_probability = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        null=True, 
        blank=True
    )
    fraud_probability_mean = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        null=True, 
        blank=True
    )
    fraud_probability_std = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        null=True, 
        blank=True
    )
    
    # Verification Results
    dl_verification_score = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        null=True, 
        blank=True
    )
    rto_verification_score = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        null=True, 
        blank=True
    )
    fir_verification_score = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        null=True, 
        blank=True
    )
    verification_reliability = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        null=True, 
        blank=True
    )
    
    # Complete Analysis Data (JSON field to store full response)
    detailed_analysis = models.JSONField(
        default=dict, 
        blank=True,
        help_text="Complete JSON response from fraud detection API"
    )
    
    # Annotated Images (store base64 or URLs)
    annotated_images = models.JSONField(
        default=list, 
        blank=True,
        help_text="List of annotated damage detection images"
    )
    
    # Admin Review
    admin_notes = models.TextField(blank=True, null=True)
    reviewed_by = models.CharField(max_length=100, blank=True, null=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    submitted_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-submitted_at']
        verbose_name = 'Claim'
        verbose_name_plural = 'Claims'
        indexes = [
            models.Index(fields=['status', '-submitted_at']),
            models.Index(fields=['policyholder', '-submitted_at']),
            models.Index(fields=['fraud_detected', 'status']),
        ]
    
    def __str__(self):
        return f"{self.claim_number} - {self.policyholder.username} - {self.status}"
    
    def save(self, *args, **kwargs):
        # Auto-generate claim number if not exists
        if not self.claim_number:
            from django.utils import timezone
            timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
            self.claim_number = f"CLM-{timestamp}-{self.policyholder.id}"
        
        # Auto-set status based on fraud detection
        if self.fraud_detected and self.status == 'Pending':
            self.status = 'Fraud'
        
        super().save(*args, **kwargs)
    
    @property
    def is_high_risk(self):
        """Check if claim is high risk"""
        return self.risk_level == 'HIGH' or self.fraud_detected
    
    @property
    def requires_review(self):
        """Check if claim requires manual review"""
        return self.status == 'Pending' or self.risk_level in ['MEDIUM', 'HIGH']


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


class ClaimHistory(models.Model):
    """Track all status changes and actions on claims"""
    ACTION_CHOICES = [
        ('submitted', 'Claim Submitted'),
        ('analyzed', 'AI Analysis Completed'),
        ('reviewed', 'Admin Reviewed'),
        ('status_changed', 'Status Changed'),
        ('approved', 'Claim Approved'),
        ('rejected', 'Claim Rejected'),
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