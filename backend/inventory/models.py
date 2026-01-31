from django.db import models
from core.models import Business, Unit, TaxRate
from django.contrib.auth.models import User

class Brand(models.Model):
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Category(models.Model):
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Product(models.Model):
    """
    Mirrors 'products' table.
    """
    TYPES = [
        ('single', 'Single'),
        ('variable', 'Variable'),
        ('combo', 'Combo')
    ]
    BARCODE_TYPES = [
        ('C128', 'Code 128'),
        ('C39', 'Code 39'),
        ('EAN13', 'EAN-13'),
        ('EAN8', 'EAN-8'),
        ('UPCA', 'UPC-A'),
        ('UPCE', 'UPC-E'),
    ]

    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    product_type = models.CharField(max_length=50, choices=TYPES, default='single')
    sku = models.CharField(max_length=255)
    
    unit = models.ForeignKey(Unit, on_delete=models.CASCADE)
    brand = models.ForeignKey(Brand, on_delete=models.SET_NULL, blank=True, null=True)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, blank=True, null=True)
    tax = models.ForeignKey(TaxRate, on_delete=models.SET_NULL, blank=True, null=True)
    tax_type = models.CharField(max_length=50, choices=[('inclusive', 'Inclusive'), ('exclusive', 'Exclusive')], default='exclusive')
    
    enable_stock = models.BooleanField(default=False)
    alert_quantity = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    
    image = models.ImageField(upload_to='products/', blank=True, null=True)
    product_description = models.TextField(blank=True, null=True)
    
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Variation(models.Model):
    """
    Mirrors 'variations' table (SKUs).
    """
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='variations')
    name = models.CharField(max_length=255, default='DUMMY') # Usually 'Dummy' for single products
    sub_sku = models.CharField(max_length=255)
    
    default_purchase_price = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    dpp_inc_tax = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    profit_percent = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    default_sell_price = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    sell_price_inc_tax = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.product.name} - {self.name}"
