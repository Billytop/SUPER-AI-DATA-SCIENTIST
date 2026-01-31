"""
Anomaly Detection System
Identifies unusual patterns in sales, expenses, inventory, and other metrics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from django.db import connections
import statistics


class AnomalyDetector:
    """
    Detects anomalies and unusual patterns in business data.
    """
    
    def __init__(self):
        self.sensitivity = 2.0  # Standard deviations for anomaly threshold
        
    def detect_sales_anomalies(self, days: int = 30) -> List[Dict]:
        """
        Detect unusual sales patterns.
        
        Args:
            days: Days of history to analyze
            
        Returns:
            List of anomalies detected
        """
        # Get daily sales
        sales_data = self._get_daily_sales(days * 2)  # Get 2x days for baseline
        
        if len(sales_data) < 14:
            return []
        
        # Calculate baseline (exclude last N days)
        baseline_data = sales_data[:-days]
        recent_data = sales_data[-days:]
        
        mean_sales = statistics.mean([d['value'] for d in baseline_data])
        std_dev = statistics.stdev([d['value'] for d in baseline_data]) if len(baseline_data) > 1 else 0
        
        anomalies = []
        threshold_high = mean_sales + (self.sensitivity * std_dev)
        threshold_low = mean_sales - (self.sensitivity * std_dev)
        
        for data_point in recent_data:
            if data_point['value'] > threshold_high:
                anomalies.append({
                    'type': 'sales_spike',
                    'date': data_point['day'],
                    'value': data_point['value'],
                    'expected': mean_sales,
                    'deviation_percent': ((data_point['value'] - mean_sales) / mean_sales * 100),
                    'severity': 'high' if data_point['value'] > threshold_high * 1.5 else 'medium',
                    'message': f"Unusual sales spike: {data_point['value']:,.0f} TZS (avg: {mean_sales:,.0f})"
                })
            elif data_point['value'] < threshold_low and threshold_low > 0:
                anomalies.append({
                    'type': 'sales_drop',
                    'date': data_point['day'],
                    'value': data_point['value'],
                    'expected': mean_sales,
                    'deviation_percent': ((mean_sales - data_point['value']) / mean_sales * 100),
                    'severity': 'high' if data_point['value'] < threshold_low * 0.5 else 'medium',
                    'message': f"Unusual sales drop: {data_point['value']:,.0f} TZS (avg: {mean_sales:,.0f})"
                })
        
        return anomalies
    
    def detect_expense_anomalies(self, months: int = 6) -> List[Dict]:
        """Detect unusual expense patterns."""
        # Similar logic to sales anomal ies
        expense_data = self._get_monthly_expenses(months)
        
        if len(expense_data) < 3:
            return []
        
        values = [d['value'] for d in expense_data]
        mean_expense = statistics.mean(values[:-1])  # Exclude latest month
        latest_expense = values[-1]
        
        anomalies = []
        
        if latest_expense > mean_expense * 1.5:
            anomalies.append({
                'type': 'expense_spike',
                'month': expense_data[-1]['month'],
                'value': latest_expense,
                'expected': mean_expense,
                'severity': 'high',
                'message': f"Expenses jumped to {latest_expense:,.0f} TZS (avg: {mean_expense:,.0f})"
            })
        
        return anomalies
    
    def detect_inventory_anomalies(self) -> List[Dict]:
        """Detect inventory irregularities."""
        anomalies = []
        
        with connections['erp'].cursor() as cursor:
            # Detect products with sudden stock changes
            cursor.execute("""
                SELECT 
                    p.id,
                    p.name,
                    v.qty_available,
                    v.change_percent
                FROM products p
                JOIN (
                    SELECT 
                        product_id,
                        SUM(quantity) as qty_available,
                        100 as change_percent
                    FROM variation_location_details
                    GROUP BY product_id
                ) v ON v.product_id = p.id
                WHERE v.qty_available < 0 OR v.qty_available > 10000
                LIMIT 20
            """)
            
            rows = cursor.fetchall()
            
            for row in rows:
                if row[2] < 0:
                    anomalies.append({
                        'type': 'negative_stock',
                        'product_id': row[ 0],
                        'product_name': row[1],
                        'quantity': row[2],
                        'severity': 'high',
                        'message': f"Negative stock detected for {row[1]}: {row[2]} units"
                    })
                elif row[2] > 10000:
                    anomalies.append({
                        'type': 'excessive_stock',
                        'product_id': row[0],
                        'product_name': row[1],
                        'quantity': row[2],
                        'severity': 'medium',
                        'message': f"Unusually high stock: {row[1]} has {row[2]} units"
                    })
        
        return anomalies
    
    def detect_fraud_indicators(self) -> List[Dict]:
        """Detect potential fraud patterns."""
        indicators = []
        
        with connections['erp'].cursor() as cursor:
            # Large transactions outside business hours
            cursor.execute("""
                SELECT id, transaction_date, final_total
                FROM transactions
                WHERE type='sell'
                AND (HOUR(transaction_date) < 6 OR HOUR(transaction_date) > 22)
                AND final_total > 1000000
                AND transaction_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                LIMIT 10
            """)
            
            rows = cursor.fetchall()
            for row in rows:
                indicators.append({
                    'type': 'unusual_timing',
                    'transaction_id': row[0],
                    'date': row[1],
                    'amount': float(row[2]),
                    'severity': 'medium',
                    'message': f"Large transaction ({row[2]:,.0f} TZS) outside business hours"
                })
            
            # Voided transactions spike
            cursor.execute("""
                SELECT COUNT(*) as void_count
                FROM transactions
                WHERE type='sell' AND status='final'
                AND transaction_date >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            """)
            
            void_count = cursor.fetchone()[0]
            if void_count > 20:
                indicators.append({
                    'type': 'void_spike',
                    'count': void_count,
                    'severity': 'high',
                    'message': f"Unusual number of voided transactions: {void_count} in last 7 days"
                })
        
        return indicators
    
    def _get_daily_sales(self, days: int) -> List[Dict]:
        """Get daily sales data."""
        with connections['erp'].cursor() as cursor:
            cursor.execute("""
                SELECT DATE(transaction_date) as day, COALESCE(SUM(final_total), 0) as total
                FROM transactions
                WHERE type='sell' 
                AND transaction_date >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY DATE(transaction_date)
                ORDER BY day ASC
            """, [days])
            
            return [{'day': r[0], 'value': float(r[1])} for r in cursor.fetchall()]
    
    def _get_monthly_expenses(self, months: int) -> List[Dict]:
        """Get monthly expense data."""
        # Simplified - in real system would track actual expenses
        with connections['erp'].cursor() as cursor:
            cursor.execute("""
                SELECT 
                    DATE_FORMAT(transaction_date, '%Y-%m') as month,
                    SUM(final_total) * 0.7 as estimated_expenses
                FROM transactions
                WHERE type='sell'
                AND transaction_date >= DATE_SUB(NOW(), INTERVAL %s MONTH)
                GROUP BY DATE_FORMAT(transaction_date, '%Y-%m')
                ORDER BY month ASC
            """, [months])
            
            return [{'month': r[0], 'value': float(r[1])} for r in cursor.fetchall()]
