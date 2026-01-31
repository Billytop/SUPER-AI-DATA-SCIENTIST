import pandas as pd
from django.db.models import Sum, Count, F
from sales.models import Transaction, TransactionSellLine
from inventory.models import Product
from django.utils import timezone
from datetime import timedelta

class KPIEngine:
    """
    Descriptive Analytics Engine.
    Uses Pandas for complex aggregations that are hard/slow in ORM.
    """
    
    @staticmethod
    def get_sales_overview(days=30):
        """
        Returns Date vs Revenue/Profit DataFrame.
        """
        start_date = timezone.now() - timedelta(days=days)
        
        # Fetch raw data needed
        queryset = Transaction.objects.filter(
            transaction_date__gte=start_date, 
            status='final'
        ).values('transaction_date', 'final_total')
        
        if not queryset.exists():
            return {"total_revenue": 0, "daily_sales": []}

        df = pd.DataFrame(list(queryset))
        df['transaction_date'] = pd.to_datetime(df['transaction_date']).dt.date
        
        # Group by Date
        daily_sales = df.groupby('transaction_date')['final_total'].sum().reset_index()
        daily_sales.columns = ['date', 'revenue']
        
        return {
            "total_revenue": float(df['final_total'].sum()),
            "daily_sales": daily_sales.to_dict('records')
        }

    @staticmethod
    def get_top_products(limit=5):
        """
        Identifies best-selling products.
        """
        queryset = TransactionSellLine.objects.values(
            name=F('product__name')
        ).annotate(
            total_qty=Sum('quantity'),
            total_revenue=Sum('line_total')
        ).order_by('-total_qty')[:limit]
        
        return list(queryset)
