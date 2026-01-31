from django.db.models.signals import post_save
from django.dispatch import receiver
from sales.models import Transaction, TransactionSellLine
from .models import Anomaly
from decimal import Decimal

@receiver(post_save, sender=Transaction)
def analyze_transaction(sender, instance, created, **kwargs):
    """
    Real-time AI Analysis of every Transaction.
    Triggers simple rule-based anomaly detection immediately.
    """
    if created:
        # 1. High Value Anomaly Check
        THRESHOLD = Decimal('100000.00')
        if instance.final_total > THRESHOLD:
            Anomaly.objects.create(
                business=instance.business,
                transaction=instance,
                description=f"High Value Transaction Detected: {instance.final_total}",
                detected_value=instance.final_total,
                expected_value=THRESHOLD,
                severity='medium',
                confidence_score=0.85
            )

        # 2. Negative Profit Check (if profit fields existed, placeholder logic)
        # if instance.final_total < instance.total_cost:
        #    Anomaly.create(...)

@receiver(post_save, sender=TransactionSellLine)
def analyze_sell_line(sender, instance, created, **kwargs):
    """
    Granular analysis of line items.
    """
    if created:
        # Check for abnormal quantity
        if instance.quantity > 1000:
             Anomaly.objects.create(
                business=instance.transaction.business,
                transaction=instance.transaction,
                description=f"Abnormal Quantity Sold: {instance.product.name} x {instance.quantity}",
                detected_value=instance.quantity,
                expected_value=100,
                severity='high',
                confidence_score=0.9
            )

from operations.models import JobSheet
from django.utils import timezone

@receiver(post_save, sender=JobSheet)
def analyze_repair_job(sender, instance, created, **kwargs):
    """
    AI Watchdog for Repair Jobs.
    """
    if not created and instance.status == 'pending':
        # Check if overdue
        if instance.delivery_date and instance.delivery_date < timezone.now():
             Anomaly.objects.create(
                business=instance.business,
                description=f"Overdue Repair Job: {instance.device} ({instance.serial_no})",
                severity='medium',
                confidence_score=0.95
            )
