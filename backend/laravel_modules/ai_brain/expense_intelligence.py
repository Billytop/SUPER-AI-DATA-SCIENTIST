"""
SephlightyAI Expense Intelligence
Author: Antigravity AI
Version: 1.0.0

Expense analysis: categorization, trend detection, anomaly detection,
cost optimization, burn rate calculation, and budget recommendations.
"""

import logging
import math
from typing import Dict, List, Any, Optional

logger = logging.getLogger("EXPENSE_INTELLIGENCE")
logger.setLevel(logging.INFO)


class ExpenseIntelligence:
    """Deep expense analytics and cost optimization engine."""

    def __init__(self):
        self._db = None
        logger.info("ExpenseIntelligence initialized.")

    def _get_db_connection(self):
        if self._db is None:
            try:
                from django.db import connections
                self._db = connections['erp_db']
            except Exception:
                from django.db import connection
                self._db = connection
        return self._db

    def _execute_query(self, sql: str, params: tuple = ()) -> List[Dict]:
        try:
            conn = self._get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"SQL Error: {e}")
            return []

    # ── Total Expenses ───────────────────────────────────────────────────

    def total_expenses(self, business_id: int,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Dict[str, Any]:
        """Calculate total expense summary."""
        where = "WHERE t.business_id = %s AND t.type = 'expense'"
        params = [business_id]

        if start_date:
            where += " AND t.transaction_date >= %s"
            params.append(start_date)
        if end_date:
            where += " AND t.transaction_date <= %s"
            params.append(end_date)

        sql = f"""
            SELECT
                COUNT(*) as total_entries,
                COALESCE(SUM(t.final_total), 0) as total_expenses,
                COALESCE(AVG(t.final_total), 0) as avg_expense,
                MIN(t.final_total) as min_expense,
                MAX(t.final_total) as max_expense
            FROM transactions t
            {where}
        """
        results = self._execute_query(sql, tuple(params))
        return results[0] if results else {}

    # ── Expense by Category ──────────────────────────────────────────────

    def expense_by_category(self, business_id: int,
                            start_date: Optional[str] = None) -> List[Dict]:
        """Breakdown of expenses by category."""
        where = "WHERE t.business_id = %s AND t.type = 'expense'"
        params = [business_id]
        if start_date:
            where += " AND t.transaction_date >= %s"
            params.append(start_date)

        sql = f"""
            SELECT
                COALESCE(ec.name, 'Uncategorized') as category,
                COUNT(*) as entries,
                SUM(t.final_total) as total,
                AVG(t.final_total) as average,
                MIN(t.transaction_date) as first_expense,
                MAX(t.transaction_date) as last_expense
            FROM transactions t
            LEFT JOIN expense_categories ec ON t.expense_category_id = ec.id
            {where}
            GROUP BY ec.id, ec.name
            ORDER BY total DESC
        """
        return self._execute_query(sql, tuple(params))

    # ── Monthly Expense Trend ────────────────────────────────────────────

    def monthly_expense_trend(self, business_id: int,
                              year: Optional[int] = None) -> List[Dict]:
        """Monthly expense trend."""
        where = "WHERE t.business_id = %s AND t.type = 'expense'"
        params = [business_id]
        if year:
            where += " AND YEAR(t.transaction_date) = %s"
            params.append(year)

        sql = f"""
            SELECT
                DATE_FORMAT(t.transaction_date, '%%Y-%%m') as month,
                COUNT(*) as entries,
                SUM(t.final_total) as total
            FROM transactions t
            {where}
            GROUP BY DATE_FORMAT(t.transaction_date, '%%Y-%%m')
            ORDER BY month
        """
        return self._execute_query(sql, tuple(params))

    # ── Anomaly Detection ────────────────────────────────────────────────

    def detect_anomalies(self, business_id: int,
                         threshold_multiplier: float = 2.0) -> List[Dict]:
        """Detect unusual expense transactions."""
        # First get average and std dev per category
        categories = self.expense_by_category(business_id)

        anomalies = []
        for cat in categories:
            avg = float(cat.get("average", 0))
            if avg <= 0:
                continue

            threshold = avg * threshold_multiplier
            category_name = cat.get("category", "Uncategorized")

            # Find expenses above threshold
            sql = """
                SELECT
                    t.id, t.transaction_date, t.final_total, t.additional_notes,
                    COALESCE(ec.name, 'Uncategorized') as category
                FROM transactions t
                LEFT JOIN expense_categories ec ON t.expense_category_id = ec.id
                WHERE t.business_id = %s AND t.type = 'expense'
                    AND t.final_total > %s
                    AND COALESCE(ec.name, 'Uncategorized') = %s
                ORDER BY t.final_total DESC
                LIMIT 5
            """
            results = self._execute_query(sql, (business_id, threshold, category_name))
            for r in results:
                r["anomaly_reason"] = f"Amount {float(r['final_total']):.0f} exceeds " \
                                      f"category avg ({avg:.0f}) by {threshold_multiplier}x"
                anomalies.append(r)

        return anomalies

    # ── Burn Rate ────────────────────────────────────────────────────────

    def calculate_burn_rate(self, business_id: int,
                            months: int = 3) -> Dict[str, Any]:
        """Calculate monthly burn rate from recent expenses."""
        monthly = self.monthly_expense_trend(business_id)

        if not monthly:
            return {"monthly_burn_rate": 0, "status": "no_data"}

        # Take last N months
        recent = monthly[-months:]
        totals = [float(m.get("total", 0)) for m in recent]

        avg_burn = sum(totals) / len(totals) if totals else 0

        # Trend
        if len(totals) >= 2:
            direction = "increasing" if totals[-1] > totals[0] else "decreasing"
        else:
            direction = "stable"

        return {
            "monthly_burn_rate": round(avg_burn, 2),
            "trend_direction": direction,
            "months_analyzed": len(recent),
            "recent_totals": totals,
        }

    # ── Cost Optimization Suggestions ────────────────────────────────────

    def optimization_suggestions(self, business_id: int) -> List[Dict]:
        """Generate cost reduction suggestions."""
        categories = self.expense_by_category(business_id)
        monthly = self.monthly_expense_trend(business_id)
        anomalies = self.detect_anomalies(business_id)

        suggestions = []

        # Suggest reviewing top expense categories
        if categories:
            top = categories[0]
            total_all = sum(float(c.get("total", 0)) for c in categories)
            top_pct = float(top.get("total", 0)) / total_all * 100 if total_all > 0 else 0

            if top_pct > 40:
                suggestions.append({
                    "area": top.get("category"),
                    "action": f"This category accounts for {top_pct:.0f}% of expenses. "
                              f"Review for possible vendor negotiation or cost reduction.",
                    "priority": "high",
                    "potential_savings": round(float(top.get("total", 0)) * 0.1, 2),
                })

        # Suggest investigating anomalies
        if anomalies:
            suggestions.append({
                "area": "anomalies",
                "action": f"{len(anomalies)} unusual expenses detected. "
                          f"Review these for errors or unauthorized spending.",
                "priority": "high",
            })

        # Suggest budget if none exists
        if monthly and len(monthly) >= 3:
            suggestions.append({
                "area": "budgeting",
                "action": "Set monthly budget targets based on 3-month averages.",
                "priority": "medium",
            })

        return suggestions

    # ── Comprehensive Report ─────────────────────────────────────────────

    def full_expense_report(self, business_id: int,
                            start_date: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive expense intelligence report."""
        return {
            "summary": self.total_expenses(business_id, start_date),
            "by_category": self.expense_by_category(business_id, start_date),
            "monthly_trend": self.monthly_expense_trend(business_id),
            "burn_rate": self.calculate_burn_rate(business_id),
            "anomalies": self.detect_anomalies(business_id),
            "optimization": self.optimization_suggestions(business_id),
        }

    # ── Quick Summary (Phase 44: SephlightyBrain Integration) ────────────

    def quick_summary(self) -> str:
        """Fast one-line expense summary — auto-detect business."""
        try:
            sql = """
                SELECT
                    COUNT(*) as entries,
                    COALESCE(SUM(final_total), 0) as total_30d
                FROM transactions
                WHERE type = 'expense'
                  AND transaction_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            """
            rows = self._execute_query(sql)
            if rows and float(rows[0].get('total_30d', 0)) > 0:
                total = float(rows[0]['total_30d'])
                entries = int(rows[0]['entries'])
                return f"💸 {entries} expenses totaling TZS {total:,.0f} in the last 30 days"
            return ""
        except Exception:
            return ""


# =============================================================================
EXPENSE_INTELLIGENCE = ExpenseIntelligence()
logger.info("Expense Intelligence v1.0.0 — ONLINE.")
