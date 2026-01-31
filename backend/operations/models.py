from django.db import models
from core.models import Business
from inventory.models import Product, Variation
from partners.models import Contact
from django.contrib.auth.models import User

class ManufacturingRecipe(models.Model):
    """
    Bill of Materials (BOM) logic.
    """
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='recipes')
    variation = models.ForeignKey(Variation, on_delete=models.CASCADE, blank=True, null=True)
    
    extra_cost = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    total_cost = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    
    ingredients = models.JSONField(default=list) # List of {product_id, quantity, unit_cost}
    instructions = models.TextField(blank=True, null=True)
    
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

class JobSheet(models.Model):
    """
    Repair Module logic.
    """
    STATUSES = [('pending', 'Pending'), ('completed', 'Completed'), ('in_progress', 'In Progress')]
    
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    contact = models.ForeignKey(Contact, on_delete=models.CASCADE) # Customer
    
    service_type = models.CharField(max_length=100, choices=[('carry_in', 'Carry In'), ('pick_up', 'Pick Up')])
    brand = models.CharField(max_length=100, blank=True, null=True)
    device = models.CharField(max_length=100, blank=True, null=True)
    serial_no = models.CharField(max_length=100, blank=True, null=True)
    security_pattern = models.CharField(max_length=100, blank=True, null=True)
    password = models.CharField(max_length=100, blank=True, null=True)
    
    problem_reported_by_customer = models.TextField(blank=True, null=True)
    
    status = models.CharField(max_length=50, choices=STATUSES, default='pending')
    delivery_date = models.DateTimeField(blank=True, null=True)
    
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

class Asset(models.Model):
    """
    Asset Management logic.
    """
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    code = models.CharField(max_length=100)
    
    purchase_date = models.DateField(blank=True, null=True)
    purchase_cost = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    
    is_allocatable = models.BooleanField(default=False)
    allocated_to = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, related_name='assets')
    
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
