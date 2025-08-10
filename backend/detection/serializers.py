from rest_framework import serializers
from django.contrib.auth.models import User
from rest_framework.permissions import AllowAny

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    permission_classes = [AllowAny]

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
