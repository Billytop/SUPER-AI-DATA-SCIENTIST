
import logging
from typing import Dict, List, Any
import datetime

logger = logging.getLogger("OMNIBRAIN_VISUALS")
logger.setLevel(logging.INFO)

class VisualizationEngine:
    """
    Sovereign Visualization Unit: Prepares Chart-ready JSON data.
    """
    def __init__(self, db_connector):
        self.db = db_connector # Reference to OmnibrainSaaSEngine for SQL execution

    def generate_customer_spending_trend(self, customer_id: int, months=6) -> Dict[str, Any]:
        """
        Line Chart: Monthly spending for a specific customer.
        """
        sql = """
            SELECT DATE_FORMAT(transaction_date, '%Y-%m') as month, SUM(final_total) as total
            FROM transactions
            WHERE contact_id = %s AND type = 'sell' AND transaction_date >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
            GROUP BY month
            ORDER BY month ASC
        """
        res = self.db._execute_erp_query(sql, (customer_id,))
        
        labels = []
        data = []
        
        # Fill missing months with 0
        current_date = datetime.date.today()
        for i in range(months-1, -1, -1):
            d = current_date - datetime.timedelta(days=30*i)
            # Simple month calculation
            year = d.year
            month = d.month
            m_str = f"{year}-{month:02d}"
            
            val = next((float(r['total']) for r in res if r['month'] == m_str), 0.0)
            labels.append(d.strftime("%b %Y"))
            data.append(val)
            
        return {
            "type": "line",
            "title": "Monthly Spending Trend",
            "labels": labels,
            "datasets": [{
                "label": "Total Spending (TZS)",
                "data": data,
                "borderColor": "#4CAF50",
                "fill": False
            }]
        }

    def generate_top_categories_pie(self, customer_id: int) -> Dict[str, Any]:
        """
        Pie Chart: Distribution of purchases by Category.
        """
        sql = """
            SELECT c.name, COUNT(*) as count
            FROM transaction_sell_lines l
            JOIN transactions t ON l.transaction_id = t.id
            JOIN variations v ON l.variation_id = v.id
            JOIN products p ON v.product_id = p.id
            JOIN categories c ON p.category_id = c.id
            WHERE t.contact_id = %s AND t.type = 'sell'
            GROUP BY c.id
            ORDER BY count DESC
            LIMIT 5
        """
        res = self.db._execute_erp_query(sql, (customer_id,))
        
        labels = [r['name'] for r in res]
        data = [r['count'] for r in res]
        
        return {
            "type": "pie",
            "title": "Favorite Categories",
            "labels": labels,
            "datasets": [{
                "data": data,
                "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
            }]
        }
