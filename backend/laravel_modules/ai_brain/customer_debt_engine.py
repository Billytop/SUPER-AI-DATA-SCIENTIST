"""
SephlightyAI Customer Debt Engine
Author: Antigravity AI
Version: 1.0.0

Per-customer ledger analysis, risk scoring, payment behavior,
credit limit advice, default prediction, and retention strategy.
"""

import logging
import math
from typing import Dict, List, Any, Optional

logger = logging.getLogger("CUSTOMER_DEBT_ENGINE")
logger.setLevel(logging.INFO)


class CustomerDebtEngine:
    """
    Deep customer intelligence: debt tracking, risk analysis, and payment behavior.
    """

    def __init__(self):
        self._db = None
        logger.info("CustomerDebtEngine initialized.")

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

    # ── Customer Ledger ──────────────────────────────────────────────────

    def customer_ledger(self, business_id: int,
                        contact_id: Optional[int] = None) -> List[Dict]:
        """Get per-customer balance sheet: sales, payments, outstanding."""
        where = "WHERE t.business_id = %s AND t.type = 'sell' AND t.status = 'final'"
        params = [business_id]

        if contact_id:
            where += " AND t.contact_id = %s"
            params.append(contact_id)

        sql = f"""
            SELECT
                ct.id as customer_id,
                ct.name as customer_name,
                ct.mobile as phone,
                COUNT(DISTINCT t.id) as total_invoices,
                COALESCE(SUM(t.final_total), 0) as total_sales,
                COALESCE(SUM(
                    (SELECT COALESCE(SUM(tp.amount), 0)
                     FROM transaction_payments tp
                     WHERE tp.transaction_id = t.id)
                ), 0) as total_paid,
                COALESCE(SUM(t.final_total), 0) -
                COALESCE(SUM(
                    (SELECT COALESCE(SUM(tp.amount), 0)
                     FROM transaction_payments tp
                     WHERE tp.transaction_id = t.id)
                ), 0) as outstanding_balance,
                MAX(t.transaction_date) as last_transaction,
                MIN(t.transaction_date) as first_transaction
            FROM transactions t
            JOIN contacts ct ON t.contact_id = ct.id
            {where}
            GROUP BY ct.id, ct.name, ct.mobile
            ORDER BY outstanding_balance DESC
        """
        return self._execute_query(sql, tuple(params))

    def top_debtors(self, business_id: int, limit: int = 10) -> List[Dict]:
        """Get customers with highest outstanding debt."""
        ledger = self.customer_ledger(business_id)
        debtors = [c for c in ledger if float(c.get("outstanding_balance", 0)) > 0]
        return debtors[:limit]

    # ── Payment Behavior ─────────────────────────────────────────────────

    def payment_behavior(self, business_id: int,
                         contact_id: int) -> Dict[str, Any]:
        """Analyze a customer's payment patterns."""
        sql = """
            SELECT
                t.id,
                t.transaction_date as invoice_date,
                t.final_total as invoice_amount,
                tp.paid_on as payment_date,
                tp.amount as payment_amount,
                tp.method as payment_method,
                DATEDIFF(tp.paid_on, t.transaction_date) as days_to_pay
            FROM transactions t
            LEFT JOIN transaction_payments tp ON tp.transaction_id = t.id
            WHERE t.business_id = %s AND t.contact_id = %s
                AND t.type = 'sell' AND t.status = 'final'
            ORDER BY t.transaction_date DESC
            LIMIT 50
        """
        payments = self._execute_query(sql, (business_id, contact_id))

        if not payments:
            return {"status": "no_data"}

        days_to_pay = [float(p["days_to_pay"]) for p in payments
                       if p.get("days_to_pay") is not None and p["days_to_pay"] >= 0]
        methods = [p["payment_method"] for p in payments if p.get("payment_method")]

        avg_days = sum(days_to_pay) / len(days_to_pay) if days_to_pay else 0
        on_time = sum(1 for d in days_to_pay if d <= 7)
        reliability = on_time / len(days_to_pay) if days_to_pay else 0

        # Method distribution
        method_counts = {}
        for m in methods:
            method_counts[m] = method_counts.get(m, 0) + 1

        return {
            "total_payments": len(payments),
            "avg_days_to_pay": round(avg_days, 1),
            "payment_reliability": round(reliability, 4),
            "on_time_payments": on_time,
            "late_payments": len(days_to_pay) - on_time,
            "preferred_method": max(method_counts, key=method_counts.get) if method_counts else "unknown",
            "method_distribution": method_counts,
        }

    # ── Risk Scoring ─────────────────────────────────────────────────────

    def calculate_risk_score(self, business_id: int,
                              contact_id: int) -> Dict[str, Any]:
        """
        Calculate a customer's risk score based on:
        - Outstanding balance ratio
        - Payment timeliness
        - Account age
        - Transaction frequency
        """
        ledger = self.customer_ledger(business_id, contact_id)
        behavior = self.payment_behavior(business_id, contact_id)

        if not ledger:
            return {"risk_score": 0.5, "risk_level": "unknown", "reason": "No data"}

        customer = ledger[0]
        total_sales = float(customer.get("total_sales", 1))
        outstanding = float(customer.get("outstanding_balance", 0))

        # Factor 1: Debt ratio (0-1, higher = riskier)
        debt_ratio = min(1.0, outstanding / total_sales) if total_sales > 0 else 0

        # Factor 2: Payment reliability (inverted: low reliability = high risk)
        reliability = behavior.get("payment_reliability", 0.5) if behavior.get("status") != "no_data" else 0.5
        reliability_risk = 1.0 - reliability

        # Factor 3: Days to pay (longer = riskier)
        avg_days = behavior.get("avg_days_to_pay", 7) if behavior.get("status") != "no_data" else 7
        days_risk = min(1.0, avg_days / 90.0)

        # Factor 4: Recency (how long since last purchase)
        # Simple proxy: more invoices = more active = lower risk
        invoices = int(customer.get("total_invoices", 0))
        activity_risk = max(0, 1.0 - invoices / 20.0)

        # Weighted risk score
        risk_score = (
            debt_ratio * 0.35 +
            reliability_risk * 0.30 +
            days_risk * 0.20 +
            activity_risk * 0.15
        )

        # Determine risk level
        if risk_score < 0.25:
            risk_level = "low"
        elif risk_score < 0.50:
            risk_level = "medium"
        elif risk_score < 0.75:
            risk_level = "high"
        else:
            risk_level = "critical"

        return {
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "factors": {
                "debt_ratio": round(debt_ratio, 4),
                "reliability_risk": round(reliability_risk, 4),
                "days_to_pay_risk": round(days_risk, 4),
                "activity_risk": round(activity_risk, 4),
            },
            "customer_name": customer.get("customer_name"),
            "outstanding": outstanding,
            "total_sales": total_sales,
        }

    # ── Credit Advice ────────────────────────────────────────────────────

    def credit_limit_advice(self, business_id: int,
                            contact_id: int) -> Dict[str, Any]:
        """Generate AI-powered credit limit recommendation."""
        risk = self.calculate_risk_score(business_id, contact_id)
        ledger = self.customer_ledger(business_id, contact_id)

        if not ledger:
            return {"recommendation": "No sufficient data", "suggested_limit": 0}

        customer = ledger[0]
        avg_order = float(customer.get("total_sales", 0)) / max(1, int(customer.get("total_invoices", 1)))
        risk_score = risk["risk_score"]

        # Base limit = 3x average order value
        base_limit = avg_order * 3

        # Adjust by risk
        if risk_score < 0.25:
            multiplier = 2.0
            recommendation = "Low risk customer. Can be extended generous credit."
        elif risk_score < 0.50:
            multiplier = 1.0
            recommendation = "Moderate risk. Standard credit terms recommended."
        elif risk_score < 0.75:
            multiplier = 0.5
            recommendation = "High risk. Reduce credit and require partial upfront payment."
        else:
            multiplier = 0.0
            recommendation = "Critical risk. Cash-only transactions recommended."

        suggested_limit = round(base_limit * multiplier, 2)

        return {
            "suggested_limit": suggested_limit,
            "recommendation": recommendation,
            "risk": risk,
            "avg_order_value": round(avg_order, 2),
        }

    # ── Comprehensive Report ─────────────────────────────────────────────

    def full_customer_report(self, business_id: int,
                              contact_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate comprehensive customer/debt intelligence report."""
        if contact_id:
            ledger = self.customer_ledger(business_id, contact_id)
            risk = self.calculate_risk_score(business_id, contact_id)
            behavior = self.payment_behavior(business_id, contact_id)
            credit = self.credit_limit_advice(business_id, contact_id)
            return {
                "ledger": ledger[0] if ledger else {},
                "risk_analysis": risk,
                "payment_behavior": behavior,
                "credit_advice": credit,
            }
        else:
            ledger = self.customer_ledger(business_id)
            debtors = self.top_debtors(business_id, 10)
            total_outstanding = sum(float(c.get("outstanding_balance", 0)) for c in ledger)
            return {
                "total_customers": len(ledger),
                "total_outstanding": total_outstanding,
                "top_debtors": debtors,
                "customers_with_debt": len([c for c in ledger if float(c.get("outstanding_balance", 0)) > 0]),
            }

    # ── Quick Summary (Phase 44: SephlightyBrain Integration) ────────────

    def quick_risk_summary(self) -> str:
        """Fast one-line debt risk overview — auto-detect business."""
        try:
            sql = """
                SELECT
                    COUNT(DISTINCT c.id) as total_customers,
                    COALESCE(SUM(t.final_total - COALESCE(paid.total_paid, 0)), 0) as total_outstanding
                FROM transactions t
                JOIN contacts c ON t.contact_id = c.id
                LEFT JOIN (
                    SELECT transaction_id, SUM(amount) as total_paid
                    FROM transaction_payments GROUP BY transaction_id
                ) paid ON paid.transaction_id = t.id
                WHERE t.type = 'sell' AND t.status = 'final'
                HAVING total_outstanding > 0
            """
            rows = self._execute_query(sql)
            if rows and rows[0].get('total_outstanding', 0) > 0:
                outstanding = float(rows[0]['total_outstanding'])
                customers = int(rows[0]['total_customers'])
                return f"⚠️ {customers} customers with TZS {outstanding:,.0f} outstanding debt"
            return ""
        except Exception:
            return ""


# =============================================================================
CUSTOMER_DEBT_ENGINE = CustomerDebtEngine()
logger.info("Customer Debt Engine v1.0.0 — ONLINE.")
