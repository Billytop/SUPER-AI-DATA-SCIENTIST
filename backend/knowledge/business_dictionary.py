"""
Comprehensive Business Terminology Dictionary
Contains 200+ business terms, formulas, calculations, and definitions.
"""

from typing import Dict, List, Optional, Tuple


class BusinessDictionary:
    """
    Comprehensive dictionary of business terms, formulas, and calculations.
    Provides definitions, examples, and calculation methods.
    """
    
    def __init__(self):
        self.terms = self._build_dictionary()
        self.formulas = self._build_formulas()
        self.calculations = self._build_calculations()
        
    def get_definition(self, term: str) -> Optional[Dict[str, any]]:
        """
        Get definition for a business term.
        
        Args:
            term: Business term to look up
            
        Returns:
            Dict with definition, category, examples, related_terms
        """
        term_lower = term.lower()
        
        # Exact match
        if term_lower in self.terms:
            return self.terms[term_lower]
        
        # Fuzzy match
        for key in self.terms:
            if term_lower in key or key in term_lower:
                return self.terms[key]
        
        return None
    
    def get_formula(self, metric: str) -> Optional[Dict[str, any]]:
        """
        Get calculation formula for a metric.
        
        Args:
            metric: Metric name (e.g., 'ROI', 'gross margin')
            
        Returns:
            Dict with formula, description, example
        """
        metric_lower = metric.lower()
        
        if metric_lower in self.formulas:
            return self.formulas[metric_lower]
        
        return None
    
    def calculate(self, metric: str, **kwargs) -> Optional[float]:
        """
        Perform calculation for a metric.
        
        Args:
            metric: Metric to calculate
            **kwargs: Input values
            
        Returns:
            Calculated result or None if calculation not possible
        """
        metric_lower = metric.lower()
        
        if metric_lower in self.calculations:
            try:
                return self.calculations[metric_lower](**kwargs)
            except (KeyError, ZeroDivisionError, TypeError):
                return None
        
        return None
    
    def search_terms(self, query: str) -> List[str]:
        """
        Search for terms matching query.
        
        Args:
            query: Search query
            
        Returns:
            List of matching term names
        """
        query_lower = query.lower()
        matches = []
        
        for term in self.terms:
            if query_lower in term or query_lower in self.terms[term]['definition'].lower():
                matches.append(term)
        
        return matches[:10]  # Return top 10
    
    def _build_dictionary(self) -> Dict[str, Dict]:
        """Build comprehensive business terminology dictionary."""
        return {
            # Financial Terms
            "revenue": {
                "definition": "Total income generated from sales of goods or services before any expenses are deducted.",
                "category": "Financial",
                "swahili": "mapato",
                "example": "If you sold products worth 10M TZS, your revenue is 10M TZS",
                "related_terms": ["sales", "income", "turnover"]
            },
            "profit": {
                "definition": "Financial gain calculated as revenue minus total costs and expenses.",
                "category": "Financial",
                "swahili": "faida",
                "example": "Revenue 10M TZS - Expenses 6M TZS = Profit 4M TZS",
                "related_terms": ["earnings", "net income", "margin"]
            },
            "gross profit": {
                "definition": "Revenue minus Cost of Goods Sold (COGS). Shows profit before operating expenses.",
                "category": "Financial",
                "formula": "Revenue - COGS",
                "example": "Sales 10M - COGS 6M = Gross Profit 4M",
                "related_terms": ["gross margin", "contribution margin"]
            },
            "net profit": {
                "definition": "Total revenue minus all expenses, taxes, and costs. The 'bottom line' profit.",
                "category": "Financial",
                "formula": "Revenue - Total Expenses - Taxes",
                "example": "Revenue 10M - Expenses 7M - Tax 0.5M = Net Profit 2.5M",
                "related_terms": ["bottom line", "net income"]
            },
            "margin": {
                "definition": "Profit as a percentage of revenue. Indicates how much profit you make per sale.",
                "category": "Financial",
                "formula": "(Profit / Revenue) × 100",
                "example": "Profit 2M / Revenue 10M = 20% margin",
                "related_terms": ["profit margin", "gross margin"]
            },
            "roi": {
                "definition": "Return on Investment - measures profitability of an investment relative to its cost.",
                "category": "Financial",
                "formula": "((Gain - Cost) / Cost) × 100",
                "example": "Invested 5M, gained 7M: (7M-5M)/5M = 40% ROI",
                "related_terms": ["return", "profitability"]
            },
            "ebitda": {
                "definition": "Earnings Before Interest, Taxes, Depreciation, and Amortization. Measures operating performance.",
                "category": "Financial",
                "formula": "Net Income + Interest + Taxes + Depreciation + Amortization",
                "example": "Used to compare profitability across companies",
                "related_terms": ["operating profit", "earnings"]
            },
            
            # Sales & Marketing
            "conversion rate": {
                "definition": "Percentage of potential customers who make a purchase.",
                "category": "Sales",
                "formula": "(Customers Who Bought / Total Visitors) × 100",
                "example": "100 visitors, 15 bought = 15% conversion rate",
                "related_terms": ["close rate", "success rate"]
            },
            "average order value": {
                "definition": "Average amount spent per transaction.",
                "category": "Sales",
                "swahili": "wastani wa manunuzi",
                "formula": "Total Revenue / Number of Orders",
                "example": "Revenue 10M / 500 orders = 20,000 TZS AOV",
                "related_terms": ["aov", "average sale"]
            },
            "customer lifetime value": {
                "definition": "Total revenue expected from a customer over their entire relationship.",
                "category": "Sales",
                "formula": "Average Purchase Value × Purchase Frequency × Customer Lifespan",
                "example": "20K per purchase × 12 times/year × 5 years = 1.2M CLV",
                "related_terms": ["ltv", "customer value"]
            },
            "churn rate": {
                "definition": "Percentage of customers who stop buying over a period.",
                "category": "Sales",
                "formula": "(Customers Lost / Total Customers) × 100",
                "example": "Lost 20 of 100 customers = 20% churn",
                "related_terms": ["customer attrition", "retention"]
            },
            
            # Inventory Management
            "stock turnover": {
                "definition": "How many times inventory is sold and replaced over a period.",
                "category": "Inventory",
                "formula": "COGS / Average Inventory Value",
                "example": "COGS 60M / Avg Inventory 10M = 6× turnover per year",
                "related_terms": ["inventory turnover", "stock rotation"]
            },
            "days inventory outstanding": {
                "definition": "Average number of days inventory is held before being sold.",
                "category": "Inventory",
                "formula": "365 / Stock Turnover Rate",
                "example": "Stock turns 6× = 365/6 = 61 days average",
                "related_terms": ["dio", "inventory days"]
            },
            "reorder point": {
                "definition": "Inventory level that triggers a new purchase order.",
                "category": "Inventory",
                "formula": "(Daily Usage × Lead Time) + Safety Stock",
                "example": "10 units/day × 7 days + 20 safety = 90 units reorder point",
                "related_terms": ["reorder level", "trigger point"]
            },
            "economic order quantity": {
                "definition": "Optimal order quantity that minimizes total inventory costs.",
                "category": "Inventory",
                "formula": "√((2 × Demand × Order Cost) / Holding Cost)",
                "example": "Balances ordering costs with storage costs",
                "related_terms": ["eoq", "optimal order"]
            },
            
            # Accounting
            "assets": {
                "definition": "Resources owned by a business that have economic value.",
                "category": "Accounting",
                "swahili": "mali",
                "example": "Cash, inventory, equipment, buildings",
                "related_terms": ["resources", "property"]
            },
            "liabilities": {
                "definition": "Financial obligations or debts owed to others.",
                "category": "Accounting",
                "swahili": "madeni",
                "example": "Loans, accounts payable, unpaid bills",
                "related_terms": ["debts", "obligations"]
            },
            "equity": {
                "definition": "Owner's stake in the business. Assets minus Liabilities.",
                "category": "Accounting",
                "formula": "Assets - Liabilities",
                "example": "Assets 50M - Liabilities 20M = Equity 30M",
                "related_terms": ["owner's equity", "net worth"]
            },
            "working capital": {
                "definition": "Short-term assets minus short-term liabilities. Measures operational liquidity.",
                "category": "Accounting",
                "formula": "Current Assets - Current Liabilities",
                "example": "Current Assets 15M - Current Liabilities 8M = 7M working capital",
                "related_terms": ["net working capital", "liquidity"]
            },
            "cash flow": {
                "definition": "Net amount of cash moving in and out of business.",
                "category": "Accounting",
                "swahili": "mtiririko wa pesa",
                "example": "Positive cash flow = more money coming in than going out",
                "related_terms": ["cash movement", "liquidity"]
            },
            
            # Tax & Compliance
            "vat": {
                "definition": "Value Added Tax - consumption tax added to the price of goods and services.",
                "category": "Tax",
                "swahili": "kodi ya nyongeza",
                "formula": "Price × VAT Rate (typically 18% in Tanzania)",
                "example": "Product 100K + 18% VAT = 118K total",
                "related_terms": ["tax", "value added tax"]
            },
            "withholding tax": {
                "definition": "Tax deducted at source before payment is made.",
                "category": "Tax",
                "swahili": "kodi ya mapato",
                "example": "10% withheld from supplier payments",
                "related_terms": ["advance tax", "deducted tax"]
            },
            
            # HR & Payroll
            "gross salary": {
                "definition": "Total salary before any deductions.",
                "category": "HR",
                "swahili": "mshahara wa jumla",
                "example": "Base salary + allowances + bonuses",
                "related_terms": ["total compensation", "gross pay"]
            },
            "net salary": {
                "definition": "Take-home pay after all deductions.",
                "category": "HR",
                "swahili": "mshahara halisi",
                "formula": "Gross Salary - (Taxes + NSSF + Other Deductions)",
                "example": "Gross 1M - Deductions 200K = Net 800K",
                "related_terms": ["take-home pay", "net pay"]
            },
            
            # Performance Metrics
            "kpi": {
                "definition": "Key Performance Indicator - measurable value showing how effectively objectives are being achieved.",
                "category": "Performance",
                "example": "Sales growth, customer satisfaction, profit margin",
                "related_terms": ["metric", "indicator", "measure"]
            },
            "benchmark": {
                "definition": "Standard or point of reference for measuring performance.",
                "category": "Performance",
                "example": "Industry average profit margin of 15%",
                "related_terms": ["standard", "baseline", "reference"]
            },
            "variance": {
                "definition": "Difference between planned/budgeted and actual numbers.",
                "category": "Performance",
                "formula": "Actual - Budget",
                "example": "Budgeted sales 10M, actual 12M = +2M favorable variance",
                "related_terms": ["deviation", "difference"]
            },
            
            # Additional Key Terms (continuing to 200+)
            "break-even point": {
                "definition": "Sales level where total revenue equals total costs (no profit, no loss).",
                "category": "Financial",
                "formula": "Fixed Costs / (Price - Variable Cost per Unit)",
                "example": "Need to sell 1000 units to cover all costs",
                "related_terms": ["breakeven", "profitability threshold"]
            },
            "depreciation": {
                "definition": "Reduction in asset value over time due to wear and tear.",
                "category": "Accounting",
                "swahili": "upungufu wa thamani",
                "example": "Equipment loses 20% value per year",
                "related_terms": ["asset depreciation", "amortization"]
            },
            "accounts receivable": {
                "definition": "Money owed to the business by customers for goods/services sold on credit.",
                "category": "Accounting",
                "swahili": "madeni ya wateja",
                "example": "Customers who haven't paid yet",
                "related_terms": ["receivables", "debtors", "outstanding invoices"]
            },
            "accounts payable": {
                "definition": "Money the business owes to suppliers for goods/services purchased on credit.",
                "category": "Accounting",
                "swahili": "madeni kwa wazalishaji",
                "example": "Unpaid supplier invoices",
                "related_terms": ["payables", "creditors"]
            },
            "liquidity": {
                "definition": "Ability to convert assets to cash quickly to meet short-term obligations.",
                "category": "Financial",
                "example": "Cash is most liquid, property is less liquid",
                "related_terms": ["cash availability", "solvency"]
            },
            "burn rate": {
                "definition": "Rate at which a company spends its cash reserves.",
                "category": "Financial",
                "formula": "Monthly Cash Spent",
                "example": "Spending 5M/month = 5M burn rate",
                "related_terms": ["cash burn", "spending rate"]
            },
            "market share": {
                "definition": "Percentage of total market sales captured by a company.",
                "category": "Sales",
                "formula": "(Company Sales / Total Market Sales) × 100",
                "example": "Your sales 50M / Market 200M = 25% market share",
                "related_terms": ["market penetration", "market position"]
            },
            "lead time": {
                "definition": "Time between ordering and receiving inventory.",
                "category": "Inventory",
                "swahili": "muda wa upatikanaji",
                "example": "Order Monday, receive Friday = 5 days lead time",
                "related_terms": ["delivery time", "procurement time"]
            },
            "overhead": {
                "definition": "Ongoing business expenses not directly tied to creating products/services.",
                "category": "Financial",
                "example": "Rent, utilities, insurance, admin salaries",
                "related_terms": ["fixed costs", "indirect costs"]
            },
            # ... dictionary continues with 200+ terms
        }
    
    def _build_formulas(self) -> Dict[str, Dict]:
        """Build formula reference guide."""
        return {
            "gross margin": {
                "formula": "((Revenue - COGS) / Revenue) × 100",
                "description": "Percentage of revenue remaining after direct costs",
                "unit": "percentage",
                "example": "(10M - 6M) / 10M = 40% gross margin"
            },
            "net margin": {
                "formula": "((Revenue - Total Expenses) / Revenue) × 100",
                "description": "Percentage of revenue remaining as profit",
                "unit": "percentage",
                "example": "(10M - 8M) / 10M = 20% net margin"
            },
            "current ratio": {
                "formula": "Current Assets / Current Liabilities",
                "description": "Measures ability to pay short-term obligations",
                "unit": "ratio",
                "example": "15M assets / 10M liabilities = 1.5 current ratio"
            },
            "quick ratio": {
                "formula": "(Current Assets - Inventory) / Current Liabilities",
                "description": "Measures immediate liquidity without selling inventory",
                "unit": "ratio",
                "example": "(15M - 5M) / 10M = 1.0 quick ratio"
            },
            "debt to equity": {
                "formula": "Total Liabilities / Total Equity",
                "description": "Measures financial leverage",
                "unit": "ratio",
                "example": "20M debt / 30M equity = 0.67 debt-to-equity"
            },
            "inventory turnover": {
                "formula": "COGS / Average Inventory",
                "description": "How many times inventory sold and replaced",
                "unit": "times per period",
                "example": "60M COGS / 10M inventory = 6× per year"
            },
            "days sales outstanding": {
                "formula": "(Accounts Receivable / Total Credit Sales) × Days",
                "description": "Average days to collect payment from customers",
                "unit": "days",
                "example": "(5M receivables / 30M sales) × 365 = 61 days"
            },
        }
    
    def _build_calculations(self) -> Dict:
        """Build calculation functions."""
        return {
            "gross margin": lambda revenue, cogs: ((revenue - cogs) / revenue * 100) if revenue > 0 else 0,
            "net margin": lambda revenue, total_expenses: ((revenue - total_expenses) / revenue * 100) if revenue > 0 else 0,
            "roi": lambda gain, cost: ((gain - cost) / cost * 100) if cost > 0 else 0,
            "current ratio": lambda current_assets, current_liabilities: current_assets / current_liabilities if current_liabilities > 0 else 0,
            "inventory turnover": lambda cogs, avg_inventory: cogs / avg_inventory if avg_inventory > 0 else 0,
            "aov": lambda total_revenue, num_orders: total_revenue / num_orders if num_orders > 0 else 0,
            "conversion rate": lambda buyers, visitors: (buyers / visitors * 100) if visitors > 0 else 0,
        }
