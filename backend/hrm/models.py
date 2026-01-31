from django.db import models
from core.models import Business
from django.contrib.auth.models import User

class Department(models.Model):
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Designation(models.Model):
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Employee(models.Model):
    """
    Extends standard User with HRM fields.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='employee_profile')
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, blank=True, null=True)
    designation = models.ForeignKey(Designation, on_delete=models.SET_NULL, blank=True, null=True)
    
    dob = models.DateField(blank=True, null=True)
    gender = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')], blank=True, null=True)
    marital_status = models.CharField(max_length=20, blank=True, null=True)
    blood_group = models.CharField(max_length=10, blank=True, null=True)
    contact_number = models.CharField(max_length=20, blank=True, null=True)
    alt_number = models.CharField(max_length=20, blank=True, null=True)
    family_number = models.CharField(max_length=20, blank=True, null=True)
    fb_link = models.URLField(blank=True, null=True)
    twitter_link = models.URLField(blank=True, null=True)
    social_media_1 = models.CharField(max_length=100, blank=True, null=True)
    social_media_2 = models.CharField(max_length=100, blank=True, null=True)
    
    permanent_address = models.TextField(blank=True, null=True)
    current_address = models.TextField(blank=True, null=True)
    
    bank_details = models.JSONField(default=dict, blank=True)
    id_proof_name = models.CharField(max_length=100, blank=True, null=True)
    id_proof_number = models.CharField(max_length=100, blank=True, null=True)
    
    joining_date = models.DateField(blank=True, null=True)
    resignation_date = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=20, default='active')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.user.get_full_name()
