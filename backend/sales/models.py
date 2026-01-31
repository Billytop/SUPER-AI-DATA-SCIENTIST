from django.db import models
from core.models import Business, TaxRate
from partners.models import Contact
from inventory.models import Product, Variation
from django.contrib.auth.models import User

class Transaction(models.Model):
    """
    The ledger of all business actions. Mirrors 'transactions' table.
    """
    TYPES = [('purchase', 'Purchase'), ('sell', 'Sell')]
    STATUSES = [
        ('received', 'Received'), 
        ('pending', 'Pending'), 
        ('ordered', 'Ordered'), 
        ('draft', 'Draft'), 
        ('final', 'Final')
    ]
    PAYMENT_STATUSES = [('paid', 'Paid'), ('due', 'Due'), ('partial', 'Partial')]

    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    location_id = models.IntegerField(blank=True, null=True) # Assuming BusinessLocation might be added later or linked to Core
    type = models.CharField(max_length=50, choices=TYPES)
    status = models.CharField(max_length=50, choices=STATUSES)
    payment_status = models.CharField(max_length=50, choices=PAYMENT_STATUSES)
    
    contact = models.ForeignKey(Contact, on_delete=models.CASCADE)
    invoice_no = models.CharField(max_length=255, blank=True, null=True)
    ref_no = models.CharField(max_length=255, blank=True, null=True)
    transaction_date = models.DateTimeField()
    
    total_before_tax = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    tax = models.ForeignKey(TaxRate, on_delete=models.SET_NULL, blank=True, null=True)
    tax_amount = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    
    discount_type = models.CharField(max_length=50, blank=True, null=True) # fixed/percentage
    discount_amount = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    
    shipping_details = models.TextField(blank=True, null=True)
    shipping_charges = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    
    final_total = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    
    # Governance & Audit
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_transactions')
    commission_agent = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, related_name='commission_transactions')
    
    is_direct_sale = models.BooleanField(default=0)
    is_suspend = models.BooleanField(default=0)
    exchange_rate = models.DecimalField(max_digits=8, decimal_places=3, default=1)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.invoice_no} ({self.type})"

class TransactionSellLine(models.Model):
    """
    Line items for a transaction. Mirrors 'transaction_sell_lines'.
    """
    transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE, related_name='sell_lines')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    variation = models.ForeignKey(Variation, on_delete=models.CASCADE)
    
    quantity = models.DecimalField(max_digits=22, decimal_places=4)
    unit_price_before_discount = models.DecimalField(max_digits=22, decimal_places=4)
    unit_price = models.DecimalField(max_digits=22, decimal_places=4)
    line_discount_type = models.CharField(max_length=50, blank=True, null=True)
    line_discount_amount = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    
    item_tax = models.DecimalField(max_digits=22, decimal_places=4, default=0)
    tax = models.ForeignKey(TaxRate, on_delete=models.SET_NULL, blank=True, null=True)
    
    unit_price_inc_tax = models.DecimalField(max_digits=22, decimal_places=4)
    
    def __str__(self):
        return f"{self.product.name} x {self.quantity}"

class TransactionPayment(models.Model):
    """
    Mirrors 'transaction_payments'.
    """
    transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE, related_name='payments')
    business = models.ForeignKey(Business, on_delete=models.CASCADE, null=True) # Added in later migrations
    amount = models.DecimalField(max_digits=22, decimal_places=4)
    method = models.CharField(max_length=50) # cash, card, etc.
    payment_ref_no = models.CharField(max_length=255, blank=True, null=True)
    paid_on = models.DateTimeField()
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
