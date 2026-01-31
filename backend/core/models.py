from django.db import models
from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey

class Business(models.Model):
    """
    Represents the main Business entity (Tenant).
    """
    name = models.CharField(max_length=255)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='businesses')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class Unit(models.Model):
    """
    Units of measurement (Piece, Kg, Litre).
    """
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    actual_name = models.CharField(max_length=255)
    short_name = models.CharField(max_length=50)
    allow_decimal = models.BooleanField(default=False)
    
    def __str__(self):
        return self.actual_name

class TaxRate(models.Model):
    """
    Tax definitions (VAT, GST).
    """
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=5, decimal_places=2) # Percentage
    is_tax_group = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.name} ({self.amount}%)"

class AuditLog(models.Model):
    """
    Tracks who did what to which object.
    """
    ACTION_CHOICES = [
        ('CREATE', 'Create'),
        ('UPDATE', 'Update'),
        ('DELETE', 'Delete'),
        ('LOGIN', 'Login'),
        ('ACCESS', 'Access'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    action = models.CharField(max_length=10, choices=ACTION_CHOICES)
    
    # Generic relation to any model
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE, null=True, blank=True)
    object_id = models.CharField(max_length=255, null=True, blank=True)
    content_object = GenericForeignKey('content_type', 'object_id')
    
    description = models.TextField(blank=True) # Details: "Changed price from 10 to 12"
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} - {self.action} - {self.description[:30]}"
