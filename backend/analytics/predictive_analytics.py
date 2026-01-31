"""
Predictive Analytics Module
Sales forecasting, inventory predictions, trend projections.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from django.db import connections
import statistics


class PredictiveAnalytics:
    """
    Forecasting and prediction capabilities for business metrics.
    """
    
    def __init__(self):
        pass
        
    def forecast_sales(self, periods: int = 30, method: str = 'moving_average') -> Dict:
        """
        Forecast future sales based on historical data.
        
        Args:
            periods: Number of days to forecast
            method: Forecasting method (moving_average, linear_trend, seasonal)
            
        Returns:
            Dict with forecast data and confidence intervals
        """
        # Get historical data
        historical_data = self._get_historical_sales(days=90)
        
        if not historical_data:
            return {'error': 'Insufficient historical data for forecasting'}
        
        if method == 'moving_average':
            forecast = self._moving_average_forecast(historical_data, periods)
        elif method == 'linear_trend':
            forecast = self._linear_trend_forecast(historical_data, periods)
        else:
            forecast = self._seasonal_forecast(historical_data, periods)
        
        return {
            'method': method,
            'forecast_periods': periods,
            'predictions': forecast,
            'confidence': self._calculate_confidence(historical_data)
        }
    
    def predict_inventory_needs(self, product_id: int, days_ahead: int = 30) -> Dict:
        """
        Predict inventory requirements for a product.
        
        Args:
            product_id: Product ID
            days_ahead: Days to predict ahead
            
        Returns:
            Dict with predicted demand and recommended order quantity
        """
        # Get historical sales rate
        sales_rate = self._get_product_sales_rate(product_id, days=60)
        
        if not sales_rate:
            return {'error': 'Insufficient sales history for this product'}
        
        # Calculate predicted demand
        predicted_demand = sales_rate * days_ahead
        
        # Add safety stock (20% buffer)
        safety_stock = predicted_demand * 0.2
        recommended_order = predicted_demand + safety_stock
        
        return {
            'product_id': product_id,
            'avg_daily_sales': sales_rate,
            'days_ahead': days_ahead,
            'predicted_demand': predicted_demand,
            'safety_stock': safety_stock,
            'recommended_order_qty': recommended_order,
            'confidence': 'medium'
        }
    
    def identify_seasonal_patterns(self, metric: str = 'sales', lookback_months: int = 12) -> Dict:
        """
        Identify seasonal patterns in business metrics.
        
        Args:
            metric: Metric to analyze (sales, inventory)
            lookback_months: Months of history to analyze
            
        Returns:
            Dict with seasonal insights
        """
        monthly_data = self._get_monthly_data(metric, lookback_months)
        
        if len(monthly_data) < 6:
            return {'error': 'Need at least 6 months of data for seasonal analysis'}
        
        # Identify peak and low months
        peak_month = max(monthly_data, key=lambda x: x['value'])
        low_month = min(monthly_data, key=lambda x: x['value'])
        
        # Calculate seasonality index
        avg_value = statistics.mean([m['value'] for m in monthly_data])
        seasonality_indices = [
            {
                'month': m['month'],
                'index': (m['value'] / avg_value) * 100
            }
            for m in monthly_data
        ]
        
        return {
            'metric': metric,
            'peak_month': peak_month['month'],
            'peak_value': peak_month['value'],
            'low_month': low_month['month'],
            'low_value': low_month['value'],
            'average': avg_value,
            'seasonality_indices': seasonality_indices,
            'pattern': self._describe_pattern(seasonality_indices)
        }
    
    def project_cash_flow(self, months_ahead: int = 3) -> Dict:
        """
        Project future cash flow based on trends.
        
        Args:
            months_ahead: Months to project
            
        Returns:
            Dict with cash flow projections
        """
        # Get historical cash flow (sales - expenses)
        historical = self._get_cash_flow_history(months=6)
        
        if not historical:
            return {'error': 'Insufficient data for projection'}
        
        # Simple linear projection
        avg_monthly = statistics.mean(historical)
        trend = self._calculate_trend(historical)
        
        projections = []
        for i in range(1, months_ahead + 1):
            projected_value = avg_monthly + (trend * i)
            projections.append({
                'month': i,
                'projected_cash_flow': projected_value,
                'confidence': 'medium' if i <= 2 else 'low'
            })
        
        return {
            'projections': projections,
            'average_monthly_historical': avg_monthly,
            'trend': 'increasing' if trend > 0 else 'decreasing',
            'recommendation': self._cash_flow_recommendation(projections)
        }
    
    def estimate_customer_lifetime_value(self, customer_id: int) -> Dict:
        """
        Estimate total value of a customer over their lifetime.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Dict with CLV estimation
        """
        with connections['erp'].cursor() as cursor:
            # Get customer stats
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT t.id) as transaction_count,
                    SUM(t.final_total) as total_spent,
                    DATEDIFF(NOW(), MIN(t.transaction_date)) as days_active,
                    AVG(t.final_total) as avg_purchase
                FROM transactions t
                WHERE t.contact_id = %s AND t.type = 'sell'
            """, [customer_id])
            
            row = cursor.fetchone()
            
            if not row or row[0] == 0:
                return {'error': 'No purchase history for this customer'}
            
            transaction_count = row[0]
            total_spent = float(row[1] or 0)
            days_active = row[2] or 1
            avg_purchase = float(row[3] or 0)
            
            # Calculate metrics
            purchases_per_year = (transaction_count / days_active) * 365 if days_active > 0 else 0
            
            # Assume 5-year customer lifetime
            estimated_lifetime_years = 5
            projected_purchases = purchases_per_year * estimated_lifetime_years
            clv = projected_purchases * avg_purchase
            
            return {
                'customer_id': customer_id,
                'historical_purchases': transaction_count,
                'total_spent_to_date': total_spent,
                'average_purchase_value': avg_purchase,
                'purchases_per_year': purchases_per_year,
                'estimated_lifetime_value': clv,
                'confidence': 'medium' if transaction_count >= 5 else 'low'
            }
    
    def _get_historical_sales(self, days: int) -> List[Dict]:
        """Get historical daily sales data."""
        with connections['erp'].cursor() as cursor:
            cursor.execute("""
                SELECT DATE(transaction_date) as day, SUM(final_total) as total
                FROM transactions
                WHERE type='sell' AND transaction_date >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY DATE(transaction_date)
                ORDER BY day ASC
            """, [days])
            
            return [{'day': r[0], 'value': float(r[1])} for r in cursor.fetchall()]
    
    def _get_product_sales_rate(self, product_id: int, days: int) -> float:
        """Get average daily sales rate for a product."""
        with connections['erp'].cursor() as cursor:
            cursor.execute("""
                SELECT SUM(sl.quantity) / %s as daily_rate
                FROM transaction_sell_lines sl
                JOIN transactions t ON t.id = sl.transaction_id
                WHERE sl.product_id = %s 
                AND t.type='sell'
                AND t.transaction_date >= DATE_SUB(NOW(), INTERVAL %s DAY)
            """, [days, product_id, days])
            
            row = cursor.fetchone()
            return float(row[0] or 0) if row else 0
    
    def _get_monthly_data(self, metric: str, months: int) -> List[Dict]:
        """Get monthly aggregated data."""
        with connections['erp'].cursor() as cursor:
            cursor.execute("""
                SELECT 
                    DATE_FORMAT(transaction_date, '%Y-%m') as month,
                    SUM(final_total) as total
                FROM transactions
                WHERE type='sell' 
                AND transaction_date >= DATE_SUB(NOW(), INTERVAL %s MONTH)
                GROUP BY DATE_FORMAT(transaction_date, '%Y-%m')
                ORDER BY month ASC
            """, [months])
            
            return [{'month': r[0], 'value': float(r[1])} for r in cursor.fetchall()]
    
    def _get_cash_flow_history(self, months: int) -> List[float]:
        """Get historical monthly cash flow."""
        # Simplified: sales - estimated expenses
        monthly_data = self._get_monthly_data('sales', months)
        return [m['value'] * 0.3 for m in monthly_data]  # Assume 30% margin
    
    def _moving_average_forecast(self, data: List[Dict], periods: int) -> List[Dict]:
        """Simple moving average forecast."""
        if not data:
            return []
        
        window_size = min(7, len(data))
        recent_values = [d['value'] for d in data[-window_size:]]
        avg = statistics.mean(recent_values)
        
        forecast = []
        for i in range(periods):
            forecast.append({
                'day': i + 1,
                'predicted_value': avg,
                'confidence': 'medium'
            })
        
        return forecast
    
    def _linear_trend_forecast(self, data: List[Dict], periods: int) -> List[Dict]:
        """Linear trend forecast."""
        if len(data) < 2:
            return []
        
        values = [d['value'] for d in data]
        trend = self._calculate_trend(values)
        last_value = values[-1]
        
        forecast = []
        for i in range(1, periods + 1):
            predicted = last_value + (trend * i)
            forecast.append({
                'day': i,
                'predicted_value': max(0, predicted),  # No negative sales
                'confidence': 'low' if i > 14 else 'medium'
            })
        
        return forecast
    
    def _seasonal_forecast(self, data: List[Dict], periods: int) -> List[Dict]:
        """Seasonal forecast (simplified)."""
        # Use last week as pattern
        pattern_length = min(7, len(data))
        pattern = [d['value'] for d in data[-pattern_length:]]
        
        forecast = []
        for i in range(periods):
            pattern_index = i % len(pattern)
            forecast.append({
                'day': i + 1,
                'predicted_value': pattern[pattern_index],
                'confidence': 'medium'
            })
        
        return forecast
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend."""
        if len(values) < 2:
            return 0
        
        n = len(values)
        x = list(range(n))
        
        # Simple linear regression
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0
    
    def _calculate_confidence(self, data: List[Dict]) -> str:
        """Calculate forecast confidence."""
        if len(data) < 30:
            return 'low'
        elif len(data) < 90:
            return 'medium'
        else:
            return 'high'
    
    def _describe_pattern(self, indices: List[Dict]) -> str:
        """Describe seasonality pattern."""
        high_months = [i['month'] for i in indices if i['index'] > 110]
        low_months = [i['month'] for i in indices if i['index'] < 90]
        
        if high_months and low_months:
            return f"Strong seasonality - peaks in {', '.join(high_months[:2])}, lows in {', '.join(low_months[:2])}"
        elif high_months:
            return f"Peaks in {', '.join(high_months[:2])}"
        elif low_months:
            return f"Lows in {', '.join(low_months[:2])}"
        else:
            return "Relatively stable throughout year"
    
    def _cash_flow_recommendation(self, projections: List[Dict]) -> str:
        """Generate cash flow recommendation."""
        avg_projection = statistics.mean([p['projected_cash_flow'] for p in projections])
        
        if avg_projection < 0:
            return "⚠️ Negative cash flow projected - urgent cost reduction or revenue increase needed"
        elif avg_projection < 1000000:
            return "💡 Positive but tight cash flow - maintain close monitoring"
        else:
            return "✅ Healthy cash flow projected - consider growth investments"
