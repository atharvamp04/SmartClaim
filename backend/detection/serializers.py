from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Policyholder

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
