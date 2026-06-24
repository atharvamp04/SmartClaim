from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Policyholder, Claim, ClaimImage, ClaimHistory, UserProfile, SurveyorFieldPhoto


# ============================================================
# EXISTING: Register & Policyholder (unchanged)
# ============================================================

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('username', 'password', 'email')

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password'],
            email=validated_data.get('email', '')
        )
        return user


class PolicyholderSerializer(serializers.ModelSerializer):
    class Meta:
        model = Policyholder
        fields = '__all__'


# ============================================================
# NEW: User Profile & Role Management
# ============================================================

class UserProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    email = serializers.EmailField(source='user.email', read_only=True)

    class Meta:
        model = UserProfile
        fields = ('username', 'email', 'role', 'employee_id', 'assigned_region', 'phone_number')


class SurveyorRegisterSerializer(serializers.ModelSerializer):
    """For admin to create surveyor accounts"""
    password = serializers.CharField(write_only=True)
    employee_id = serializers.CharField(write_only=True, required=False, allow_blank=True)
    assigned_region = serializers.CharField(write_only=True, required=False, allow_blank=True)
    phone_number = serializers.CharField(write_only=True, required=False, allow_blank=True)

    class Meta:
        model = User
        fields = ('username', 'password', 'email', 'employee_id', 'assigned_region', 'phone_number')

    def create(self, validated_data):
        # Extract profile fields before creating user
        employee_id = validated_data.pop('employee_id', '')
        assigned_region = validated_data.pop('assigned_region', '')
        phone_number = validated_data.pop('phone_number', '')

        user = User.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password'],
            email=validated_data.get('email', '')
        )

        # Create UserProfile with surveyor role
        UserProfile.objects.create(
            user=user,
            role='surveyor',
            employee_id=employee_id,
            assigned_region=assigned_region,
            phone_number=phone_number
        )

        return user


class LoginResponseSerializer(serializers.Serializer):
    """Serializer for login response including role"""
    access = serializers.CharField()
    refresh = serializers.CharField()
    role = serializers.CharField()
    username = serializers.CharField()
    employee_id = serializers.CharField(allow_null=True, required=False)


# ============================================================
# NEW: Surveyor List (for admin dropdown)
# ============================================================

class SurveyorListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing surveyors in admin panel"""
    username = serializers.CharField(source='user.username')
    email = serializers.EmailField(source='user.email')
    active_claims = serializers.SerializerMethodField()

    class Meta:
        model = UserProfile
        fields = ('username', 'email', 'employee_id', 'assigned_region', 'phone_number', 'active_claims')

    def get_active_claims(self, obj):
        return obj.user.assigned_claims.filter(status='Under Survey').count()


# ============================================================
# EXISTING + MODIFIED: ClaimImage
# ============================================================

class ClaimImageSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()
    annotated_image_url = serializers.SerializerMethodField()
    damage_areas = serializers.JSONField(source='damage_regions')

    class Meta:
        model = ClaimImage
        fields = (
            'id', 'image_index', 'image_url', 'annotated_image_url',
            'fraud_probability', 'confidence', 'damage_percentage',
            'severity_level', 'damage_areas_count', 'damage_areas',
            'uploaded_at'
        )

    def get_image_url(self, obj):
        request = self.context.get('request')
        if obj.image_file and request:
            return request.build_absolute_uri(obj.image_file.url)
        return obj.image_file.url if obj.image_file else None

    def get_annotated_image_url(self, obj):
        request = self.context.get('request')
        if obj.annotated_image and request:
            return request.build_absolute_uri(obj.annotated_image.url)
        return obj.annotated_image.url if obj.annotated_image else None


# ============================================================
# NEW: Surveyor Field Photo
# ============================================================

class SurveyorFieldPhotoSerializer(serializers.ModelSerializer):
    photo_url = serializers.SerializerMethodField()
    uploaded_by_username = serializers.CharField(source='uploaded_by.username', read_only=True)

    class Meta:
        model = SurveyorFieldPhoto
        fields = ('id', 'photo_url', 'uploaded_by_username', 'caption', 'uploaded_at')

    def get_photo_url(self, obj):
        request = self.context.get('request')
        if obj.photo and request:
            return request.build_absolute_uri(obj.photo.url)
        return obj.photo.url if obj.photo else None


# ============================================================
# EXISTING + MODIFIED: ClaimHistory
# ============================================================

class ClaimHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ClaimHistory
        fields = ('id', 'action', 'old_status', 'new_status', 'performed_by', 'notes', 'timestamp')


# ============================================================
# NEW: Claim Timeline for Visual Step Display
# ============================================================

class ClaimTimelineStepSerializer(serializers.Serializer):
    """
    Represents a single step in the claim timeline.
    Maps ClaimHistory records to visual timeline steps.
    """
    step = serializers.CharField()  # e.g., "Submitted", "AI Analysis", etc.
    status = serializers.CharField()  # Current claim status at this step
    timestamp = serializers.DateTimeField()
    performed_by = serializers.CharField()
    notes = serializers.CharField(required=False, allow_blank=True)
    is_completed = serializers.BooleanField()


class ClaimTimelineSerializer(serializers.Serializer):
    """
    Complete timeline for a claim showing all steps from submission to decision.
    """
    claim_id = serializers.IntegerField()
    claim_number = serializers.CharField()
    current_status = serializers.CharField()
    submitted_at = serializers.DateTimeField()
    timeline_steps = ClaimTimelineStepSerializer(many=True)
    
    class Meta:
        fields = ('claim_id', 'claim_number', 'current_status', 'submitted_at', 'timeline_steps')


# ============================================================
# EXISTING + MODIFIED: Claim (surveyor fields added)
# ============================================================

class ClaimListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for claim lists/tables"""
    policyholder = serializers.SerializerMethodField()
    assigned_surveyor_name = serializers.SerializerMethodField()

    class Meta:
        model = Claim
        fields = (
            'id', 'claim_number', 'policyholder', 'claim_amount',
            'confidence_score', 'risk_level', 'status', 'fraud_detected',
            'submitted_at', 'assigned_surveyor_name',
            # NEW surveyor fields
            'surveyor_recommendation', 'damage_verified', 'surveyor_assessed_amount',
        )

    def get_policyholder(self, obj):
        return {
            'name': obj.policyholder.username,
            'email': obj.policyholder.email,
            'username': obj.policyholder.username,
        }

    def get_assigned_surveyor_name(self, obj):
        return obj.assigned_surveyor.username if obj.assigned_surveyor else None


class ClaimDetailSerializer(serializers.ModelSerializer):
    """Full serializer for claim detail page"""
    policyholder = PolicyholderSerializer(read_only=True)
    images = ClaimImageSerializer(many=True, read_only=True)
    history = ClaimHistorySerializer(many=True, read_only=True)
    field_photos = SurveyorFieldPhotoSerializer(many=True, read_only=True)  # NEW

    # Computed fields
    assigned_surveyor_name = serializers.SerializerMethodField()
    fraud_explanation = serializers.JSONField(default=dict)
    fraud_summary = serializers.CharField(read_only=True, allow_null=True)
    final_claim_amount = serializers.DecimalField(
        max_digits=10, decimal_places=2, read_only=True
    )
    is_assigned_to_surveyor = serializers.BooleanField(read_only=True)

    class Meta:
        model = Claim
        fields = (
            # Basic info
            'id', 'claim_number', 'claim_description', 'accident_date',
            'claim_amount', 'status', 'risk_level', 'fraud_detected',
            'submitted_at', 'updated_at',

            # Documents
            'dl_number', 'vehicle_reg_no', 'fir_number',

            # Images
            'total_images_submitted', 'images', 'field_photos',

            # Fraud scores
            'confidence_score', 'tabular_fraud_probability',
            'image_fraud_probability', 'fusion_score',
            'fraud_explanation', 'fraud_summary',

            # Damage info
            'overall_damage_severity', 'total_damage_areas',
            'average_damage_percentage', 'max_damage_percentage',

            # Verification
            'dl_verification_score', 'rto_verification_score',
            'fir_verification_score', 'verification_reliability',

            # Admin
            'admin_notes', 'reviewed_by', 'reviewed_at',

            # NEW: Surveyor fields
            'assigned_surveyor_name', 'assigned_at',
            'surveyor_notes', 'surveyor_recommendation',
            'surveyor_assessed_amount', 'damage_verified',
            'survey_completed_at', 'final_claim_amount',
            'is_assigned_to_surveyor',

            # Relations
            'policyholder', 'history',
        )

    def get_assigned_surveyor_name(self, obj):
        return obj.assigned_surveyor.username if obj.assigned_surveyor else None


# ============================================================
# NEW: Surveyor-specific claim serializer
# ============================================================

class SurveyorClaimSerializer(serializers.ModelSerializer):
    """
    Serializer for surveyor's view of claims.
    Excludes sensitive internal fraud scores — surveyors
    should do an unbiased field inspection.
    """
    policyholder_name = serializers.CharField(source='policyholder.username', read_only=True)
    policyholder_email = serializers.EmailField(source='policyholder.email', read_only=True)
    vehicle_info = serializers.SerializerMethodField()
    images = ClaimImageSerializer(many=True, read_only=True)
    field_photos = SurveyorFieldPhotoSerializer(many=True, read_only=True)
    history = ClaimHistorySerializer(many=True, read_only=True)

    class Meta:
        model = Claim
        fields = (
            'id', 'claim_number', 'claim_description', 'accident_date',
            'claim_amount', 'status', 'risk_level', 'submitted_at',

            # Documents
            'dl_number', 'vehicle_reg_no', 'fir_number',

            # Policyholder info
            'policyholder_name', 'policyholder_email', 'vehicle_info',

            # Images
            'total_images_submitted', 'images', 'field_photos',

            # Damage info (visible to surveyor)
            'overall_damage_severity', 'total_damage_areas',
            'average_damage_percentage',

            # Surveyor's own report fields
            'surveyor_notes', 'surveyor_recommendation',
            'surveyor_assessed_amount', 'damage_verified',
            'survey_completed_at', 'assigned_at',

            'history',
        )

    def get_vehicle_info(self, obj):
        return {
            'make': obj.policyholder.vehicle_make,
            'model': obj.policyholder.vehicle_model,
            'category': obj.policyholder.vehicle_category,
            'year': obj.policyholder.year_of_vehicle,
            'age': obj.policyholder.age_of_vehicle,
        }


# ============================================================
# NEW: Survey Report submission serializer
# ============================================================

class SurveyReportSerializer(serializers.Serializer):
    """Validates data when surveyor submits a field report"""
    survey_notes = serializers.CharField(min_length=10)
    recommendation = serializers.ChoiceField(
        choices=['APPROVE', 'REJECT', 'INVESTIGATE', 'PARTIAL']
    )
    actual_damage_amount = serializers.DecimalField(
        max_digits=10, decimal_places=2,
        required=False, allow_null=True
    )
    damage_verified = serializers.BooleanField(default=True)


# ============================================================
# NEW: Assign Surveyor serializer
# ============================================================

class AssignSurveyorSerializer(serializers.Serializer):
    """Validates data when admin assigns a surveyor to a claim"""
    surveyor_username = serializers.CharField()

    def validate_surveyor_username(self, value):
        try:
            user = User.objects.get(username=value)
            if not hasattr(user, 'profile') or user.profile.role != 'surveyor':
                raise serializers.ValidationError(
                    f"User '{value}' is not a surveyor."
                )
        except User.DoesNotExist:
            raise serializers.ValidationError(
                f"User '{value}' does not exist."
            )
        return value