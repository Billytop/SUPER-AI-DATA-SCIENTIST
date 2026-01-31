"""
Intelligent Recommendation System
Product recommendations, customer targeting, pricing suggestions, upsell opportunities.
"""

from typing import Dict, List, Tuple
from django.db import connections


class Recommender:
    """
    Generates intelligent business recommendations based on data analysis.
    """
    
    def __init__(self):
        pass
        
    def recommend_products_for_customer(self, customer_id: int, limit: int = 5) -> List[Dict]:
        """
        Recommend products for a specific customer based on purchase history.
        
        Args:
            customer_id: Customer ID
            limit: Number of recommendations
            
        Returns:
            List of product recommendations
        """
        with connections['erp'].cursor() as cursor:
            # Get customer's previously purchased products
            cursor.execute("""
                SELECT DISTINCT p.category_id
                FROM transaction_sell_lines sl
               JOIN products p ON p.id = sl.product_id
                JOIN transactions t ON t.id = sl.transaction_id
                WHERE t.contact_id = %s AND t.type = 'sell'
            """, [customer_id])
            
            category_ids = [r[0] for r in cursor.fetchall()]
            
            if not category_ids:
                return self._recommend_popular_products(limit)
            
            # Find popular products in those categories that customer hasn't bought
            placeholders = ','.join(['%s'] * len(category_ids))
            cursor.execute(f"""
                SELECT 
                    p.id,
                    p.name,
                    COUNT(DISTINCT t.id) as purchase_count,
                    AVG(sl.unit_price_inc_tax) as avg_price
                FROM products p
                JOIN transaction_sell_lines sl ON sl.product_id = p.id
                JOIN transactions t ON t.id = sl.transaction_id
                WHERE p.category_id IN ({placeholders})
                AND p.id NOT IN (
                    SELECT DISTINCT sl2.product_id
                    FROM transaction_sell_lines sl2
                    JOIN transactions t2 ON t2.id = sl2.transaction_id
                    WHERE t2.contact_id = %s
                )
                GROUP BY p.id, p.name
                ORDER BY purchase_count DESC
                LIMIT %s
            """, category_ids + [customer_id, limit])
            
            recommendations = []
            for row in cursor.fetchall():
                recommendations.append({
                    'product_id': row[0],
                    'product_name': row[1],
                    'popularity_score': row[2],
                    'price': float(row[3]),
                    'reason': 'Popular in categories you like'
                })
            
            return recommendations
    
    def recommend_upsell_products(self, product_id: int, limit: int = 3) -> List[Dict]:
        """
        Recommend higher-value alternatives or complementary products.
        
        Args:
            product_id: Base product ID
            limit: Number of recommendations
            
        Returns:
            List of upsell recommendations
        """
        with connections['erp'].cursor() as cursor:
            # Get product details
            cursor.execute("""
                SELECT category_id, unit_price_inc_tax
                FROM products
                WHERE id = %s
            """, [product_id])
            
            row = cursor.fetchone()
            if not row:
                return []
            
            category_id = row[0]
            base_price = float(row[1] or 0)
            
            # Find higher-priced products in same category
            cursor.execute("""
                SELECT 
                    p.id,
                    p.name,
                    v.default_sell_price as price,
                    ((v.default_sell_price - %s) / %s * 100) as price_increase
                FROM products p
                JOIN variations v ON v.product_id = p.id
                WHERE p.category_id = %s
                AND p.id != %s
                AND v.default_sell_price > %s
                AND v.default_sell_price < %s * 1.5
                ORDER BY v.default_sell_price ASC
                LIMIT %s
            """, [base_price, base_price, category_id, product_id, base_price, base_price, limit])
            
            recommendations = []
            for row in cursor.fetchall():
                recommendations.append({
                    'product_id': row[0],
                    'product_name': row[1],
                    'price': float(row[2]),
                    'price_increase_percent': float(row[3]),
                    'reason': 'Premium alternative with better features'
                })
            
            return recommendations
    
    def recommend_bundle_products(self, product_id: int, limit: int = 3) -> List[Dict]:
        """
        Recommend products frequently bought together.
        
        Args:
            product_id: Base product ID
            limit: Number of recommendations
            
        Returns:
            List of bundle recommendations
        """
        with connections['erp'].cursor() as cursor:
            # Find products bought in same transactions
            cursor.execute("""
                SELECT 
                    p2.id,
                    p2.name,
                    COUNT(DISTINCT t.id) as co_purchase_count,
                    AVG(sl2.unit_price_inc_tax) as avg_price
                FROM transactions t
                JOIN transaction_sell_lines sl1 ON sl1.transaction_id = t.id
                JOIN transaction_sell_lines sl2 ON sl2.transaction_id = t.id
                JOIN products p2 ON p2.id = sl2.product_id
                WHERE sl1.product_id = %s
                AND sl2.product_id != %s
                AND t.type = 'sell'
                GROUP BY p2.id, p2.name
                HAVING co_purchase_count >= 2
                ORDER BY co_purchase_count DESC
                LIMIT %s
            """, [product_id, product_id, limit])
            
            recommendations = []
            for row in cursor.fetchall():
                recommendations.append({
                    'product_id': row[0],
                    'product_name': row[1],
                    'bundle_score': row[2],
                    'price': float(row[3]),
                    'reason': f'Frequently bought together ({row[2]} times)'
                })
            
            return recommendations
    
    def recommend_pricing_strategy(self, product_id: int) -> Dict:
        """
        Suggest optimal pricing based on sales data.
        
        Args:
            product_id: Product ID
            
        Returns:
            Dict with pricing recommendation
        """
        with connections['erp'].cursor() as cursor:
            # Get sales at different price points
            cursor.execute("""
                SELECT 
                    sl.unit_price_inc_tax as price,
                    SUM(sl.quantity) as units_sold,
                    COUNT(DISTINCT t.id) as transactions
                FROM transaction_sell_lines sl
                JOIN transactions t ON t.id = sl.transaction_id
                WHERE sl.product_id = %s
                AND t.type = 'sell'
                AND t.transaction_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
                GROUP BY sl.unit_price_inc_tax
                ORDER BY units_sold DESC
            """, [product_id])
            
            price_points = []
            for row in cursor.fetchall():
                price_points.append({
                    'price': float(row[0]),
                    'units_sold': row[1],
                    'transactions': row[2],
                    'revenue': float(row[0]) * row[1]
                })
            
            if not price_points:
                return {'error': 'Insufficient sales data for pricing recommendation'}
            
            # Find optimal price (highest revenue)
            optimal = max(price_points, key=lambda x: x['revenue'])
            current = price_points[0]  # Most recent
            
            recommendation = {
                'current_price': current['price'],
                'recommended_price': optimal['price'],
                'expected_revenue_increase': optimal['revenue'] - current['revenue'],
                'price_points_analyzed': len(price_points),
                'reasoning': f"Price of {optimal['price']:,.0f} TZS generated highest revenue in last 90 days"
            }
            
            if optimal['price'] > current['price']:
                recommendation['action'] = 'increase_price'
                recommendation['message'] = f"Consider increasing price to {optimal['price']:,.0f} TZS for optimal revenue"
            elif optimal['price'] < current['price']:
                recommendation['action'] = 'decrease_price'
                recommendation['message'] = f"Consider decreasing price to {optimal['price']:,.0f} TZS to boost sales volume"
            else:
                recommendation['action'] = 'maintain_price'
                recommendation['message'] = "Current pricing is optimal"
            
            return recommendation
    
    def recommend_restocking_priority(self, limit: int = 10) -> List[Dict]:
        """
        Prioritize which products to restock based on sales velocity and stock levels.
        
        Args:
            limit: Number of recommendations
            
        Returns:
            List of restocking recommendations
        """
        with connections['erp'].cursor() as cursor:
            cursor.execute("""
                SELECT 
                    p.id,
                    p.name,
                    COALESCE(v.qty_available, 0) as current_stock,
                    COALESCE(sales.daily_rate, 0) as daily_sales_rate,
                    COALESCE(v.qty_available / NULLIF(sales.daily_rate, 0), 0) as days_of_stock
                FROM products p
                LEFT JOIN (
                    SELECT product_id, SUM(quantity) as qty_available
                    FROM variation_location_details
                    GROUP BY product_id
                ) v ON v.product_id = p.id
                LEFT JOIN (
                    SELECT 
                        sl.product_id,
                        SUM(sl.quantity) / 30 as daily_rate
                    FROM transaction_sell_lines sl
                    JOIN transactions t ON t.id = sl.transaction_id
                    WHERE t.type = 'sell'
                    AND t.transaction_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                    GROUP BY sl.product_id
                ) sales ON sales.product_id = p.id
                WHERE sales.daily_rate > 0
                ORDER BY days_of_stock ASC
                LIMIT %s
            """, [limit])
            
            recommendations = []
            for row in cursor.fetchall():
                days_remaining = row[4] if row[4] is not None else 0
                
                if days_remaining < 7:
                    priority = 'urgent'
                    message = 'Restock immediately - will run out in < 7 days'
                elif days_remaining < 14:
                    priority = 'high'
                    message = 'Restock soon - will run out in < 2 weeks'
                else:
                    priority = 'medium'
                    message = 'Monitor stock levels'
                
                recommendations.append({
                    'product_id': row[0],
                    'product_name': row[1],
                    'current_stock': row[2],
                    'daily_sales_rate': float(row[3]),
                    'days_remaining': days_remaining,
                    'priority': priority,
                    'message': message
                })
            
            return recommendations
    
    def recommend_customer_targets(self, campaign_type: str = 'reactivation', limit: int = 20) -> List[Dict]:
        """
        Recommend customers to target for marketing campaigns.
        
        Args:
            campaign_type: Type of campaign (reactivation, upsell, loyalty)
            limit: Number of customers
            
        Returns:
            List of customer targets
        """
        with connections['erp'].cursor() as cursor:
            if campaign_type == 'reactivation':
                # Customers who haven't purchased recently but were active before
                cursor.execute("""
                    SELECT 
                        c.id,
                        c.name,
                        MAX(t.transaction_date) as last_purchase,
                        DATEDIFF(NOW(), MAX(t.transaction_date)) as days_since_purchase,
                        SUM(t.final_total) as lifetime_value
                    FROM contacts c
                    JOIN transactions t ON t.contact_id = c.id
                    WHERE c.type = 'customer'
                    AND t.type = 'sell'
                    GROUP BY c.id, c.name
                    HAVING days_since_purchase BETWEEN 60 AND 180
                    AND lifetime_value > 100000
                    ORDER BY lifetime_value DESC
                    LIMIT %s
                """, [limit])
                
            elif campaign_type == 'upsell':
                # High-value customers with recent activity
                cursor.execute("""
                    SELECT 
                        c.id,
                        c.name,
                        MAX(t.transaction_date) as last_purchase,
                        SUM(t.final_total) as lifetime_value,
                        AVG(t.final_total) as avg_purchase
                    FROM contacts c
                    JOIN transactions t ON t.contact_id = c.id
                    WHERE c.type = 'customer'
                    AND t.type = 'sell'
                    GROUP BY c.id, c.name
                    HAVING MAX(t.transaction_date) >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                    AND lifetime_value > 500000
                    ORDER BY lifetime_value DESC
                    LIMIT %s
                """, [limit])
                
            else:  # loyalty
                # Most loyal customers
                cursor.execute("""
                    SELECT 
                        c.id,
                        c.name,
                        COUNT(DISTINCT t.id) as purchase_count,
                        SUM(t.final_total) as lifetime_value,
                        DATEDIFF(NOW(), MIN(t.transaction_date)) as customer_age_days
                    FROM contacts c
                    JOIN transactions t ON t.contact_id = c.id
                    WHERE c.type = 'customer'
                    AND t.type = 'sell'
                    GROUP BY c.id, c.name
                    HAVING purchase_count >= 5
                    ORDER BY purchase_count DESC, lifetime_value DESC
                    LIMIT %s
                """, [limit])
            
            customers = []
            for row in cursor.fetchall():
                customers.append({
                    'customer_id': row[0],
                    'customer_name': row[1],
                    'campaign_type': campaign_type,
                    'score': row[3] if len(row) > 3 else 0,
                    'reason': self._get_campaign_reason(campaign_type, row)
                })
            
            return customers
    
    def _recommend_popular_products(self, limit: int) -> List[Dict]:
        """Fallback: recommend generally popular products."""
        with connections['erp'].cursor() as cursor:
            cursor.execute("""
                SELECT 
                    p.id,
                    p.name,
                    COUNT(DISTINCT t.id) as purchase_count,
                    AVG(sl.unit_price_inc_tax) as avg_price
                FROM products p
                JOIN transaction_sell_lines sl ON sl.product_id = p.id
                JOIN transactions t ON t.id = sl.transaction_id
                WHERE t.type = 'sell'
                AND t.transaction_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
                GROUP BY p.id, p.name
                ORDER BY purchase_count DESC
                LIMIT %s
            """, [limit])
            
            return [{
                'product_id': r[0],
                'product_name': r[1],
                'popularity_score': r[2],
                'price': float(r[3]),
                'reason': 'Popular product'
            } for r in cursor.fetchall()]
    
    def _get_campaign_reason(self, campaign_type: str, row: Tuple) -> str:
        """Generate campaign reason message."""
        if campaign_type == 'reactivation':
            return f"High-value customer (LTV: {row[4]:,.0f} TZS) inactive for {row[3]} days"
        elif campaign_type == 'upsell':
            return f"Active high-value customer (LTV: {row[3]:,.0f} TZS)"
        else:
            return f"Loyal customer with {row[2]} purchases over {row[4]} days"
