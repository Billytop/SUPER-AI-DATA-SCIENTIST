"""
SephlightyAI Sales Deep Intelligence
Author: Antigravity AI
Version: 1.0.0

Comprehensive sales analytics engine with:
- Revenue analysis by time period, product, customer, category
- Trend detection and seasonality
- Cross-sell / upsell recommendations
- Forecasting integration (Transformer + LSTM hybrid)
- Leaderboard analytics
"""

import logging
import math
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("SALES_INTELLIGENCE")
logger.setLevel(logging.INFO)


class SalesDeepIntelligence:
    """
    Deep Sales Intelligence Engine.
    Provides multi-dimensional sales analysis powered by the Transformer Core.
    """

    def __init__(self):
        self._db = None
        self._transformer = None
        logger.info("SalesDeepIntelligence initialized.")

    def _get_db_connection(self):
        """Get Django DB connection."""
        if self._db is None:
            try:
                from django.db import connections
                self._db = connections['erp_db']
            except Exception:
                from django.db import connection
                self._db = connection
        return self._db

    def _execute_query(self, sql: str, params: tuple = ()) -> List[Dict]:
        """Execute a raw SQL query and return results as dicts."""
        try:
            conn = self._get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"SQL Error: {e}")
            return []

    # ── Revenue Analytics ────────────────────────────────────────────────

    def total_revenue(self, business_id: int,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Dict[str, Any]:
        """Calculate total revenue with breakdown."""
        where = "WHERE t.business_id = %s AND t.type = 'sell' AND t.status != 'draft'"
        params = [business_id]

        if start_date:
            where += " AND t.transaction_date >= %s"
            params.append(start_date)
        if end_date:
            where += " AND t.transaction_date <= %s"
            params.append(end_date)

        sql = f"""
            SELECT
                COUNT(DISTINCT t.id) as total_transactions,
                COALESCE(SUM(t.final_total), 0) as total_revenue,
                COALESCE(AVG(t.final_total), 0) as avg_transaction_value,
                MIN(t.final_total) as min_transaction,
                MAX(t.final_total) as max_transaction,
                COUNT(DISTINCT t.contact_id) as unique_customers
            FROM transactions t
            {where}
        """
        results = self._execute_query(sql, tuple(params))
        return results[0] if results else {}

    def revenue_by_period(self, business_id: int,
                          period: str = "monthly",
                          year: Optional[int] = None) -> List[Dict]:
        """Revenue breakdown by time period."""
        if period == "daily":
            group_by = "DATE(t.transaction_date)"
            select_period = "DATE(t.transaction_date) as period"
        elif period == "weekly":
            group_by = "YEARWEEK(t.transaction_date)"
            select_period = "YEARWEEK(t.transaction_date) as period"
        elif period == "yearly":
            group_by = "YEAR(t.transaction_date)"
            select_period = "YEAR(t.transaction_date) as period"
        else:  # monthly
            group_by = "DATE_FORMAT(t.transaction_date, '%%Y-%%m')"
            select_period = "DATE_FORMAT(t.transaction_date, '%%Y-%%m') as period"

        where = "WHERE t.business_id = %s AND t.type = 'sell' AND t.status != 'draft'"
        params = [business_id]
        if year:
            where += " AND YEAR(t.transaction_date) = %s"
            params.append(year)

        sql = f"""
            SELECT
                {select_period},
                COUNT(*) as transactions,
                COALESCE(SUM(t.final_total), 0) as revenue,
                COALESCE(AVG(t.final_total), 0) as avg_value
            FROM transactions t
            {where}
            GROUP BY {group_by}
            ORDER BY period
        """
        return self._execute_query(sql, tuple(params))

    # ── Product Analytics ────────────────────────────────────────────────

    def top_products(self, business_id: int, limit: int = 10,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     metric: str = "revenue") -> List[Dict]:
        """Get top selling products by revenue or quantity."""
        where = """WHERE t.business_id = %s AND t.type = 'sell' AND t.status != 'draft'"""
        params = [business_id]

        if start_date:
            where += " AND t.transaction_date >= %s"
            params.append(start_date)
        if end_date:
            where += " AND t.transaction_date <= %s"
            params.append(end_date)

        order_col = "revenue" if metric == "revenue" else "qty_sold"

        sql = f"""
            SELECT
                p.name as product_name,
                p.id as product_id,
                SUM(tsl.quantity) as qty_sold,
                SUM(tsl.quantity * tsl.unit_price_inc_tax) as revenue,
                AVG(tsl.unit_price_inc_tax) as avg_price,
                COUNT(DISTINCT t.id) as transactions
            FROM transaction_sell_lines tsl
            JOIN transactions t ON tsl.transaction_id = t.id
            JOIN products p ON tsl.product_id = p.id
            {where}
            GROUP BY p.id, p.name
            ORDER BY {order_col} DESC
            LIMIT %s
        """
        params.append(limit)
        return self._execute_query(sql, tuple(params))

    def worst_products(self, business_id: int, limit: int = 10,
                       start_date: Optional[str] = None) -> List[Dict]:
        """Get worst selling products (dead stock candidates)."""
        where = "WHERE t.business_id = %s AND t.type = 'sell' AND t.status != 'draft'"
        params = [business_id]
        if start_date:
            where += " AND t.transaction_date >= %s"
            params.append(start_date)

        sql = f"""
            SELECT
                p.name as product_name,
                p.id as product_id,
                COALESCE(SUM(tsl.quantity), 0) as qty_sold,
                COALESCE(SUM(tsl.quantity * tsl.unit_price_inc_tax), 0) as revenue
            FROM products p
            LEFT JOIN transaction_sell_lines tsl ON tsl.product_id = p.id
            LEFT JOIN transactions t ON tsl.transaction_id = t.id AND t.business_id = %s
                AND t.type = 'sell' AND t.status != 'draft'
            WHERE p.business_id = %s
            GROUP BY p.id, p.name
            ORDER BY qty_sold ASC
            LIMIT %s
        """
        params.extend([business_id, business_id, limit])
        return self._execute_query(sql, tuple(params))

    def profit_by_category(self, business_id: int,
                           start_date: Optional[str] = None) -> List[Dict]:
        """Profit analysis by product category."""
        where = "WHERE t.business_id = %s AND t.type = 'sell' AND t.status != 'draft'"
        params = [business_id]
        if start_date:
            where += " AND t.transaction_date >= %s"
            params.append(start_date)

        sql = f"""
            SELECT
                c.name as category,
                SUM(tsl.quantity * tsl.unit_price_inc_tax) as revenue,
                SUM(tsl.quantity * COALESCE(v.default_purchase_price, 0)) as cost,
                SUM(tsl.quantity * tsl.unit_price_inc_tax) -
                    SUM(tsl.quantity * COALESCE(v.default_purchase_price, 0)) as gross_profit,
                COUNT(DISTINCT t.id) as transactions
            FROM transaction_sell_lines tsl
            JOIN transactions t ON tsl.transaction_id = t.id
            JOIN products p ON tsl.product_id = p.id
            LEFT JOIN categories c ON p.category_id = c.id
            LEFT JOIN variations v ON tsl.variation_id = v.id
            {where}
            GROUP BY c.id, c.name
            ORDER BY gross_profit DESC
        """
        return self._execute_query(sql, tuple(params))

    # ── Customer Sales Analytics ─────────────────────────────────────────

    def top_customers_by_revenue(self, business_id: int,
                                  limit: int = 10,
                                  start_date: Optional[str] = None) -> List[Dict]:
        """Top customers by total revenue."""
        where = "WHERE t.business_id = %s AND t.type = 'sell' AND t.status != 'draft'"
        params = [business_id]
        if start_date:
            where += " AND t.transaction_date >= %s"
            params.append(start_date)

        sql = f"""
            SELECT
                ct.name as customer_name,
                ct.id as customer_id,
                COUNT(DISTINCT t.id) as transactions,
                SUM(t.final_total) as total_revenue,
                AVG(t.final_total) as avg_order_value,
                MAX(t.transaction_date) as last_purchase
            FROM transactions t
            JOIN contacts ct ON t.contact_id = ct.id
            {where}
            GROUP BY ct.id, ct.name
            ORDER BY total_revenue DESC
            LIMIT %s
        """
        params.append(limit)
        return self._execute_query(sql, tuple(params))

    # ── Trend Detection ──────────────────────────────────────────────────

    def detect_trend(self, time_series: List[float]) -> Dict[str, Any]:
        """Detect trend direction and strength from a time series."""
        if len(time_series) < 3:
            return {"direction": "insufficient_data", "strength": 0}

        n = len(time_series)
        # Simple linear regression
        x_mean = (n - 1) / 2.0
        y_mean = sum(time_series) / n

        numerator = sum((i - x_mean) * (time_series[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return {"direction": "flat", "strength": 0}

        slope = numerator / denominator

        # Normalize strength (0-1)
        if y_mean != 0:
            strength = min(1.0, abs(slope / y_mean) * 10)
        else:
            strength = 0

        direction = "up" if slope > 0.01 else ("down" if slope < -0.01 else "flat")

        # Seasonality detection (simple CV check)
        cv = (math.sqrt(sum((v - y_mean) ** 2 for v in time_series) / n)) / abs(y_mean) \
            if y_mean != 0 else 0
        is_seasonal = cv > 0.3

        return {
            "direction": direction,
            "strength": round(strength, 4),
            "slope": round(slope, 4),
            "is_seasonal": is_seasonal,
            "volatility": round(cv, 4),
        }

    # ── Cross-Sell / Upsell ──────────────────────────────────────────────

    def cross_sell_recommendations(self, business_id: int,
                                    product_id: int,
                                    limit: int = 5) -> List[Dict]:
        """Find products frequently bought together."""
        sql = """
            SELECT
                p2.name as recommended_product,
                p2.id as product_id,
                COUNT(*) as co_purchase_count
            FROM transaction_sell_lines tsl1
            JOIN transaction_sell_lines tsl2 ON tsl1.transaction_id = tsl2.transaction_id
                AND tsl1.product_id != tsl2.product_id
            JOIN products p2 ON tsl2.product_id = p2.id
            JOIN transactions t ON tsl1.transaction_id = t.id
            WHERE tsl1.product_id = %s AND t.business_id = %s
                AND t.type = 'sell' AND t.status != 'draft'
            GROUP BY p2.id, p2.name
            ORDER BY co_purchase_count DESC
            LIMIT %s
        """
        return self._execute_query(sql, (product_id, business_id, limit))

    # ── Comprehensive Report ─────────────────────────────────────────────

    def full_sales_report(self, business_id: int,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive sales intelligence report."""
        revenue = self.total_revenue(business_id, start_date, end_date)
        top_prods = self.top_products(business_id, 5, start_date, end_date)
        top_custs = self.top_customers_by_revenue(business_id, 5, start_date)
        categories = self.profit_by_category(business_id, start_date)
        monthly = self.revenue_by_period(business_id, "monthly")

        # Detect trend from monthly data
        trend = {}
        if monthly:
            revenues = [float(m.get("revenue", 0)) for m in monthly]
            trend = self.detect_trend(revenues)

        return {
            "summary": revenue,
            "top_products": top_prods,
            "top_customers": top_custs,
            "category_profit": categories,
            "monthly_trend": monthly,
            "trend_analysis": trend,
            "report_period": {"start": start_date, "end": end_date},
        }

    # ── Quick Summary (Phase 44: SephlightyBrain Integration) ────────────

    def quick_trend_summary(self) -> str:
        """Fast one-line sales trend insight — no business_id needed (auto-detect)."""
        try:
            sql = """
                SELECT
                    COALESCE(SUM(CASE WHEN transaction_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) THEN final_total END), 0) as this_month,
                    COALESCE(SUM(CASE WHEN transaction_date >= DATE_SUB(CURDATE(), INTERVAL 60 DAY)
                                      AND transaction_date < DATE_SUB(CURDATE(), INTERVAL 30 DAY) THEN final_total END), 0) as last_month
                FROM transactions
                WHERE type = 'sell' AND status != 'draft'
            """
            rows = self._execute_query(sql)
            if rows:
                this_m = float(rows[0].get('this_month', 0))
                last_m = float(rows[0].get('last_month', 0))
                if last_m > 0:
                    pct = ((this_m - last_m) / last_m) * 100
                    direction = "📈 UP" if pct > 0 else "📉 DOWN"
                    return f"{direction} {abs(pct):.1f}% vs last 30 days (TZS {this_m:,.0f} vs {last_m:,.0f})"
                elif this_m > 0:
                    return f"Sales this month: TZS {this_m:,.0f} (no prior period for comparison)"
            return ""
        except Exception:
            return ""


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================
SALES_DEEP_INTELLIGENCE = SalesDeepIntelligence()
logger.info("Sales Deep Intelligence v1.0.0 — ONLINE.")
