from django.db import models
from core.models import Business
from django.contrib.auth.models import User

class Contact(models.Model):
    """
    Mirrors 'contacts' table (Customers & Suppliers).
    """
    TYPES = [
        ('customer', 'Customer'),
        ('supplier', 'Supplier'),
        ('both', 'Both')
    ]
    STATUSES = [
        ('active', 'Active'),
        ('inactive', 'Inactive')
    ]

    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='contacts')
    type = models.CharField(max_length=50, choices=TYPES)
    supplier_business_name = models.CharField(max_length=255, blank=True, null=True)
    name = models.CharField(max_length=255)
    email = models.EmailField(blank=True, null=True)
    mobile = models.CharField(max_length=50)
    tax_number = models.CharField(max_length=100, blank=True, null=True)
    pay_term_number = models.IntegerField(blank=True, null=True)
    pay_term_type = models.CharField(max_length=50, blank=True, null=True) # days/months
    credit_limit = models.DecimalField(max_digits=22, decimal_places=4, blank=True, null=True)
    contact_status = models.CharField(max_length=50, choices=STATUSES, default='active')
    
    # Financials
    balance = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.type})"

class CustomerGroup(models.Model):
    """
    Mirrors 'customer_groups' table.
    """
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=22, decimal_places=4) # Discount amount
    price_calculation_type = models.CharField(max_length=50, default='percentage') # percentage/selling_price_group
    
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name
