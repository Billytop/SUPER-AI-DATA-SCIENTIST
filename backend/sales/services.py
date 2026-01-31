from decimal import Decimal, ROUND_HALF_UP
from django.core.exceptions import ValidationError
from core.models import TaxRate, Business
from sales.models import Transaction, TransactionSellLine
from inventory.models import Variation

class TransactionService:
    """
    Ports logic from IKO SAWA 'TransactionUtil.php'
    """
    
    @staticmethod
    def validate_transaction_date(business_id, date):
        # Placeholder for Fiscal Year Lock
        pass

    @staticmethod
    def calculate_tax(business_id, total_before_tax, tax_rate_id, submitted_tax):
        """
        Implements Bank-Grade Tax Governance Check.
        Mirrors lines 53-72 of TransactionUtil.php
        """
        business = Business.objects.get(id=business_id)
        mode = business.common_settings.get('compliance_mode', 'sme')
        
        if mode in ['bank', 'plc'] and tax_rate_id:
            tax_rate = TaxRate.objects.get(id=tax_rate_id)
            
            # High precision math
            rate_fraction = tax_rate.amount / Decimal('100.00')
            expected_tax = total_before_tax * rate_fraction
            
            # Quantize to 4 decimals
            expected_tax = expected_tax.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
            submitted_tax = Decimal(str(submitted_tax)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
            
            diff = abs(expected_tax - submitted_tax)
            
            if diff > Decimal('0.05'):
                raise ValidationError(
                    f"Regulatory Check Failed: Manual tax override not allowed in {mode} mode. "
                    f"Expected: {expected_tax}, Got: {submitted_tax}"
                )
        return True

    @staticmethod
    def check_stock_governance(transaction: Transaction, requested_qty, variation_id, location_id):
        """
        Implements Governance Logic for Negative Stock (Anti-Overselling).
        Mirrors lines 337-373 of TransactionUtil.php
        """
        if transaction.status == 'final':
            business_details = transaction.business
            pos_settings = business_details.pos_settings
            allow_overselling = pos_settings.get('allow_overselling', False)
            
            if not allow_overselling:
                # In real implementation we would check VariationLocationDetails
                # For now simulating logic
                current_stock = Decimal('100.00') # Placeholder
                if requested_qty > current_stock:
                     prod_name = Variation.objects.get(id=variation_id).product.name
                     raise ValidationError(
                         f"Governance Error: Insufficient Stock for {prod_name}. "
                         f"Available: {current_stock}. Requested: {requested_qty}. Overselling is BLOCKED."
                     )
        return True
