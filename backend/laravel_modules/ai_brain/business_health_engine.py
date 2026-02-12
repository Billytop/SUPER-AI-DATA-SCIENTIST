"""
SephlightyAI Business Health Engine
Author: Antigravity AI
Version: 1.0.0

Holistic business health analysis:
- P&L (Profit & Loss) calculation
- Cash flow analysis
- Liquidity ratios
- Survival analysis (runway estimation)
- Growth rate tracking
- Overall health score (A-F grade)
"""

import logging
import math
from typing import Dict, List, Any, Optional

logger = logging.getLogger("BUSINESS_HEALTH")
logger.setLevel(logging.INFO)


class BusinessHealthEngine:
    """
    Comprehensive business health assessment engine.
    Aggregates data from sales, expenses, inventory, and customers
    to produce a holistic health score.
    """

    def __init__(self):
        self._db = None
        self._sales = None
        self._expenses = None
        self._debt = None
        logger.info("BusinessHealthEngine initialized.")

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

    # ── Lazy-load sub-engines ────────────────────────────────────────────

    def _get_sales_engine(self):
        if self._sales is None:
            try:
                from laravel_modules.ai_brain.sales_deep_intelligence import SALES_DEEP_INTELLIGENCE
                self._sales = SALES_DEEP_INTELLIGENCE
            except ImportError:
                pass
        return self._sales

    def _get_expense_engine(self):
        if self._expenses is None:
            try:
                from laravel_modules.ai_brain.expense_intelligence import EXPENSE_INTELLIGENCE
                self._expenses = EXPENSE_INTELLIGENCE
            except ImportError:
                pass
        return self._expenses

    def _get_debt_engine(self):
        if self._debt is None:
            try:
                from laravel_modules.ai_brain.customer_debt_engine import CUSTOMER_DEBT_ENGINE
                self._debt = CUSTOMER_DEBT_ENGINE
            except ImportError:
                pass
        return self._debt

    # ── Profit & Loss ────────────────────────────────────────────────────

    def profit_and_loss(self, business_id: int,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Dict[str, Any]:
        """Calculate Profit & Loss statement."""
        # Revenue
        revenue_sql = """
            SELECT COALESCE(SUM(final_total), 0) as total_revenue
            FROM transactions
            WHERE business_id = %s AND type = 'sell' AND status = 'final'
        """
        params = [business_id]
        if start_date:
            revenue_sql += " AND transaction_date >= %s"
            params.append(start_date)
        if end_date:
            revenue_sql += " AND transaction_date <= %s"
            params.append(end_date)

        revenue_result = self._execute_query(revenue_sql, tuple(params))
        total_revenue = float(revenue_result[0].get("total_revenue", 0)) if revenue_result else 0

        # Cost of Goods Sold
        cogs_sql = """
            SELECT COALESCE(SUM(final_total), 0) as total_cogs
            FROM transactions
            WHERE business_id = %s AND type = 'purchase' AND status = 'received'
        """
        cogs_params = [business_id]
        if start_date:
            cogs_sql += " AND transaction_date >= %s"
            cogs_params.append(start_date)
        if end_date:
            cogs_sql += " AND transaction_date <= %s"
            cogs_params.append(end_date)

        cogs_result = self._execute_query(cogs_sql, tuple(cogs_params))
        total_cogs = float(cogs_result[0].get("total_cogs", 0)) if cogs_result else 0

        # Operating Expenses
        expense_sql = """
            SELECT COALESCE(SUM(final_total), 0) as total_expenses
            FROM transactions
            WHERE business_id = %s AND type = 'expense'
        """
        exp_params = [business_id]
        if start_date:
            expense_sql += " AND transaction_date >= %s"
            exp_params.append(start_date)
        if end_date:
            expense_sql += " AND transaction_date <= %s"
            exp_params.append(end_date)

        exp_result = self._execute_query(expense_sql, tuple(exp_params))
        total_expenses = float(exp_result[0].get("total_expenses", 0)) if exp_result else 0

        # Computations
        gross_profit = total_revenue - total_cogs
        net_profit = gross_profit - total_expenses
        gross_margin = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
        net_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0

        return {
            "total_revenue": round(total_revenue, 2),
            "cost_of_goods_sold": round(total_cogs, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_margin_pct": round(gross_margin, 2),
            "operating_expenses": round(total_expenses, 2),
            "net_profit": round(net_profit, 2),
            "net_margin_pct": round(net_margin, 2),
            "is_profitable": net_profit > 0,
        }

    # ── Cash Flow ────────────────────────────────────────────────────────

    def cash_flow_summary(self, business_id: int,
                          start_date: Optional[str] = None) -> Dict[str, Any]:
        """Summarize cash inflows and outflows."""
        # Cash inflows (payments received)
        inflow_sql = """
            SELECT COALESCE(SUM(tp.amount), 0) as total_inflow
            FROM transaction_payments tp
            JOIN transactions t ON tp.transaction_id = t.id
            WHERE t.business_id = %s AND t.type = 'sell'
        """
        params = [business_id]
        if start_date:
            inflow_sql += " AND tp.paid_on >= %s"
            params.append(start_date)

        inflow = self._execute_query(inflow_sql, tuple(params))
        total_inflow = float(inflow[0].get("total_inflow", 0)) if inflow else 0

        # Cash outflows (purchases + expenses paid)
        outflow_sql = """
            SELECT COALESCE(SUM(tp.amount), 0) as total_outflow
            FROM transaction_payments tp
            JOIN transactions t ON tp.transaction_id = t.id
            WHERE t.business_id = %s AND t.type IN ('purchase', 'expense')
        """
        out_params = [business_id]
        if start_date:
            outflow_sql += " AND tp.paid_on >= %s"
            out_params.append(start_date)

        outflow = self._execute_query(outflow_sql, tuple(out_params))
        total_outflow = float(outflow[0].get("total_outflow", 0)) if outflow else 0

        net_cash = total_inflow - total_outflow

        return {
            "cash_inflow": round(total_inflow, 2),
            "cash_outflow": round(total_outflow, 2),
            "net_cash_flow": round(net_cash, 2),
            "is_positive": net_cash > 0,
        }

    # ── Survival Analysis ────────────────────────────────────────────────

    def survival_analysis(self, business_id: int) -> Dict[str, Any]:
        """
        Estimate business runway:
        How many months can the business survive at current burn rate?
        """
        # Get current cash reserves (net cash flow to date)
        cash = self.cash_flow_summary(business_id)
        net_cash = cash.get("net_cash_flow", 0)

        # Get burn rate
        expense_engine = self._get_expense_engine()
        if expense_engine:
            burn = expense_engine.calculate_burn_rate(business_id)
            monthly_burn = burn.get("monthly_burn_rate", 0)
        else:
            monthly_burn = 0

        if monthly_burn > 0 and net_cash > 0:
            runway_months = net_cash / monthly_burn
        else:
            runway_months = float('inf') if net_cash >= 0 else 0

        # Status assessment
        if runway_months == float('inf'):
            status = "healthy"
            assessment = "Business has positive cash flow with no burn concern."
        elif runway_months >= 12:
            status = "stable"
            assessment = f"Business has ~{runway_months:.0f} months of runway."
        elif runway_months >= 6:
            status = "caution"
            assessment = f"Only {runway_months:.0f} months of runway. Consider cost cuts."
        elif runway_months >= 3:
            status = "warning"
            assessment = f"Critical: Only {runway_months:.0f} months of runway remaining."
        else:
            status = "danger"
            assessment = f"Emergency: Less than {max(0, runway_months):.0f} months remaining!"

        return {
            "net_cash_reserve": round(net_cash, 2),
            "monthly_burn_rate": round(monthly_burn, 2),
            "runway_months": round(runway_months, 1) if runway_months != float('inf') else "unlimited",
            "status": status,
            "assessment": assessment,
        }

    # ── Growth Rate ──────────────────────────────────────────────────────

    def growth_rate(self, business_id: int,
                    period: str = "monthly") -> Dict[str, Any]:
        """Calculate revenue growth rate."""
        sales_engine = self._get_sales_engine()
        if not sales_engine:
            return {"status": "no_data"}

        monthly = sales_engine.revenue_by_period(business_id, period)
        if len(monthly) < 2:
            return {"status": "insufficient_data"}

        revenues = [float(m.get("revenue", 0)) for m in monthly]
        periods_list = [m.get("period", "") for m in monthly]

        # Month-over-month growth rates
        growth_rates = []
        for i in range(1, len(revenues)):
            if revenues[i - 1] > 0:
                rate = (revenues[i] - revenues[i - 1]) / revenues[i - 1] * 100
                growth_rates.append(round(rate, 2))

        avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0

        # Latest growth
        latest_growth = growth_rates[-1] if growth_rates else 0

        return {
            "avg_growth_rate": round(avg_growth, 2),
            "latest_growth_rate": latest_growth,
            "growth_rates": growth_rates[-12:],
            "periods": periods_list[-12:],
            "is_growing": avg_growth > 0,
        }

    # ── Overall Health Score ─────────────────────────────────────────────

    def health_score(self, business_id: int) -> Dict[str, Any]:
        """
        Calculate overall business health score (0-100) and grade (A-F).

        Factors:
        1. Profitability (30%)
        2. Cash flow (25%)
        3. Revenue growth (20%)
        4. Debt health (15%)
        5. Expense management (10%)
        """
        scores = {}

        # 1. Profitability score
        pnl = self.profit_and_loss(business_id)
        net_margin = pnl.get("net_margin_pct", 0)
        if net_margin >= 20:
            scores["profitability"] = 100
        elif net_margin >= 10:
            scores["profitability"] = 80
        elif net_margin >= 0:
            scores["profitability"] = 60
        elif net_margin >= -10:
            scores["profitability"] = 30
        else:
            scores["profitability"] = 10

        # 2. Cash flow score
        cash = self.cash_flow_summary(business_id)
        if cash.get("is_positive"):
            scores["cash_flow"] = 90
        elif cash.get("net_cash_flow", 0) >= 0:
            scores["cash_flow"] = 60
        else:
            scores["cash_flow"] = 20

        # 3. Growth score
        growth = self.growth_rate(business_id)
        avg_growth = growth.get("avg_growth_rate", 0)
        if avg_growth >= 10:
            scores["growth"] = 100
        elif avg_growth >= 5:
            scores["growth"] = 80
        elif avg_growth >= 0:
            scores["growth"] = 60
        elif avg_growth >= -5:
            scores["growth"] = 30
        else:
            scores["growth"] = 10

        # 4. Debt health score
        debt_engine = self._get_debt_engine()
        if debt_engine:
            report = debt_engine.full_customer_report(business_id)
            total_outstanding = report.get("total_outstanding", 0)
            total_revenue = pnl.get("total_revenue", 1)
            debt_ratio = total_outstanding / total_revenue if total_revenue > 0 else 0

            if debt_ratio < 0.1:
                scores["debt_health"] = 100
            elif debt_ratio < 0.3:
                scores["debt_health"] = 70
            elif debt_ratio < 0.5:
                scores["debt_health"] = 40
            else:
                scores["debt_health"] = 10
        else:
            scores["debt_health"] = 50

        # 5. Expense management score
        survival = self.survival_analysis(business_id)
        status = survival.get("status", "unknown")
        if status == "healthy":
            scores["expense_mgmt"] = 100
        elif status == "stable":
            scores["expense_mgmt"] = 80
        elif status == "caution":
            scores["expense_mgmt"] = 50
        elif status == "warning":
            scores["expense_mgmt"] = 25
        else:
            scores["expense_mgmt"] = 10

        # Weighted overall score
        weights = {
            "profitability": 0.30,
            "cash_flow": 0.25,
            "growth": 0.20,
            "debt_health": 0.15,
            "expense_mgmt": 0.10,
        }
        overall = sum(scores[k] * weights[k] for k in weights)

        # Grade
        if overall >= 90:
            grade = "A"
        elif overall >= 80:
            grade = "B"
        elif overall >= 65:
            grade = "C"
        elif overall >= 50:
            grade = "D"
        else:
            grade = "F"

        return {
            "overall_score": round(overall, 1),
            "grade": grade,
            "factor_scores": scores,
            "weights": weights,
            "pnl_summary": pnl,
            "cash_flow_summary": cash,
            "growth_summary": growth,
            "survival_summary": survival,
        }

    # ── Full Health Report ───────────────────────────────────────────────

    def full_health_report(self, business_id: int,
                           start_date: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive business health report."""
        return {
            "health_score": self.health_score(business_id),
            "profit_and_loss": self.profit_and_loss(business_id, start_date),
            "cash_flow": self.cash_flow_summary(business_id, start_date),
            "growth": self.growth_rate(business_id),
            "survival": self.survival_analysis(business_id),
        }

    # ── Quick Summary (Phase 44: SephlightyBrain Integration) ────────────

    def quick_health_summary(self) -> str:
        """Fast one-line business health summary — auto-detect business."""
        try:
            # Quick P&L snapshot
            sql_rev = """
                SELECT COALESCE(SUM(final_total), 0) as rev
                FROM transactions
                WHERE type = 'sell' AND status = 'final'
                  AND transaction_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            """
            sql_exp = """
                SELECT COALESCE(SUM(final_total), 0) as exp
                FROM transactions
                WHERE type = 'expense'
                  AND transaction_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            """
            rev_rows = self._execute_query(sql_rev)
            exp_rows = self._execute_query(sql_exp)
            revenue = float(rev_rows[0]['rev']) if rev_rows else 0
            expenses = float(exp_rows[0]['exp']) if exp_rows else 0
            profit = revenue - expenses
            if revenue > 0:
                margin = (profit / revenue) * 100
                grade = "A" if margin >= 20 else "B" if margin >= 10 else "C" if margin >= 0 else "D"
                status = "profitable ✅" if profit > 0 else "at a loss ⚠️"
                return f"Grade {grade} — {status} (margin {margin:.1f}%, revenue TZS {revenue:,.0f})"
            return ""
        except Exception:
            return ""


# =============================================================================
BUSINESS_HEALTH_ENGINE = BusinessHealthEngine()
logger.info("Business Health Engine v1.0.0 — ONLINE.")
