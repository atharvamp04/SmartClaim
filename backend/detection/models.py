from django.db import models

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
