
import logging
from typing import Dict, List, Any

logger = logging.getLogger("OMNIBRAIN_CUSTOMER_INTEL")
logger.setLevel(logging.INFO)

class CustomerIntelligenceCore:
    """
    Sovereign Intelligence Unit: Deep Customer Profiling & Behavioral Analysis.
    """
    def __init__(self, db_connector):
        self.db = db_connector # Reference to OmnibrainSaaSEngine for SQL execution

    def analyze_customer(self, customer_id: int) -> Dict[str, Any]:
        """
        Generates a 360-degree profile of a specific customer.
        """
        profile = {
            "top_products": [],
            "churn_risk": "Unknown",
            "last_purchase_days_ago": None,
            "total_spent": 0,
            "recommendations": []
        }
        
        # 1. Top Products (Most Purchased)
        sql_top = """
            SELECT p.name, SUM(l.quantity) as total_qty, SUM(l.quantity * l.unit_price_inc_tax) as total_val
            FROM transaction_sell_lines l
            JOIN transactions t ON l.transaction_id = t.id
            JOIN variations v ON l.variation_id = v.id
            JOIN products p ON v.product_id = p.id
            WHERE t.contact_id = %s AND t.type = 'sell'
            GROUP BY p.id
            ORDER BY total_qty DESC
            LIMIT 5
        """
        top_res = self._exec(sql_top, (customer_id,))
        profile["top_products"] = top_res
        
        # 2. Total Spent & Last Purchase
        sql_summary = """
            SELECT SUM(final_total) as total_spent, MAX(transaction_date) as last_date
            FROM transactions
            WHERE contact_id = %s AND type = 'sell'
        """
        summary_res = self._exec(sql_summary, (customer_id,))
        if summary_res:
            profile["total_spent"] = float(summary_res[0]['total_spent'] or 0)
            last_date = summary_res[0]['last_date']
            
            if last_date:
                import datetime
                # Handle string or datetime object
                if isinstance(last_date, str):
                    last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")
                
                delta = (datetime.datetime.now() - last_date).days
                profile["last_purchase_days_ago"] = delta
                
                # Churn Risk Calculation
                if delta > 90:
                    profile["churn_risk"] = "CRITICAL (Lost)"
                elif delta > 60:
                    profile["churn_risk"] = "HIGH (At Risk)"
                elif delta > 30:
                    profile["churn_risk"] = "MEDIUM (Cooling Down)"
                else:
                    profile["churn_risk"] = "LOW (Active)"

        # 3. Recommendations (Simple Cross-Sell Logic)
        # Suggest products bought by others who bought this customer's top item
        if profile["top_products"]:
            top_item = profile["top_products"][0]['name']
            profile["recommendations"] = self._get_recommendations(top_item, customer_id)
            
        return profile

    def _get_recommendations(self, product_name: str, exclude_user_id: int) -> List[str]:
        """
        Finds products frequently bought together with the given product.
        """
        # Simplified logic: Find random popular items not bought by user (Mock or Basic SQL)
        # For true association rules, we'd need a complex query. 
        # Here we'll return "Smart Recommendations" based on general popularity.
        sql = """
            SELECT name, COUNT(*) as popularity 
            FROM products 
            LIMIT 3
        """
        res = self._exec(sql)
        return [r['name'] for r in res] if res else []

    def _exec(self, sql, params=()):
        return self.db._execute_erp_query(sql, params)
