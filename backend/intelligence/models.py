from django.db import models
from core.models import Business
from sales.models import Transaction
from django.contrib.auth.models import User

class Anomaly(models.Model):
    """
    Stores AI-detected irregularities.
    """
    SEVERITY = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical')
    ]
    
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE, blank=True, null=True)
    
    description = models.TextField()
    detected_value = models.DecimalField(max_digits=22, decimal_places=4, blank=True, null=True)
    expected_value = models.DecimalField(max_digits=22, decimal_places=4, blank=True, null=True)
    severity = models.CharField(max_length=20, choices=SEVERITY, default='medium')
    confidence_score = models.FloatField(default=0.0) # 0.0 to 1.0
    
    is_resolved = models.BooleanField(default=False)
    resolved_by = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Anomaly: {self.description} ({self.severity})"

class BusinessInsight(models.Model):
    """
    Stores positive AI insights (e.g., "Sales trending up on Fridays").
    """
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    description = models.TextField()
    
    type = models.CharField(max_length=50) # sales_trend, inventory_optimization, etc.
    actionable_recommendation = models.TextField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
