"""
Industry-Specific Knowledge Base
ERP best practices, tax regulations, business strategies for Tanzania.
"""

from typing import Dict, List, Optional


class KnowledgeBase:
    """
    Domain expertise knowledge base for ERP, accounting, tax, and business operations.
    """
    
    def __init__(self):
        self.erp_best_practices = self._build_erp_practices()
        self.tax_knowledge = self._build_tax_knowledge()
        self.inventory_strategies = self._build_inventory_strategies()
        self.sales_optimization = self._build_sales_optimization()
        self.financial_planning = self._build_financial_planning()
        
    def get_best_practice(self, topic: str) -> Optional[Dict]:
        """Get best practice for a specific topic."""
        topic_lower = topic.lower()
        
        all_practices = {
            **self.erp_best_practices,
            **self.inventory_strategies,
            **self.sales_optimization,
            **self.financial_planning
        }
        
        for key, practice in all_practices.items():
            if topic_lower in key:
                return practice
        
        return None
    
    def get_tax_info(self, tax_type: str) -> Optional[Dict]:
        """Get tax information."""
        tax_lower = tax_type.lower()
        
        for key, info in self.tax_knowledge.items():
            if tax_lower in key:
                return info
        
        return None
    
    def _build_erp_practices(self) -> Dict:
        """ERP implementation and usage best practices."""
        return {
            "data_entry": {
                "title": "Consistent Data Entry",
                "description": "Maintain consistency in how you enter data across the system",
                "tips": [
                    "Use standardized product naming conventions",
                    "Set up product categories before adding products",
                    "Enforce consistent customer name formats",
                    "Use dropdown lists where possible to avoid typos"
                ],
                "benefits": ["Better reporting", "Easier searching", "Reduced errors"]
            },
            "inventory_management": {
                "title": "Effective Inventory Management",
                "description": "Optimize stock levels and reduce carrying costs",
                "tips": [
                    "Set reorder points for all products",
                    "Conduct regular stock counts",
                    "Use ABC analysis for inventory prioritization",
                    "Monitor slow-moving items monthly"
                ],
                "benefits": ["Reduced stockouts", "Lower carrying costs", "Better cash flow"]
            },
            "customer_relations": {
                "title": "Customer Relationship Management",
                "description": "Build and maintain strong customer relationships",
                "tips": [
                    "Track customer payment history",
                    "Set credit limits based on payment behavior",
                    "Follow up on overdue invoices promptly",
                    "Reward loyal customers with special pricing"
                ],
                "benefits": ["Better collections", "Increased loyalty", "Higher sales"]
            },
            "reporting": {
                "title": "Regular Financial Reporting",
                "description": "Generate and review key reports regularly",
                "tips": [
                    "Review daily sales report every morning",
                    "Check weekly debt aging report",
                    "Analyze monthly P&L statement",
                    "Review quarterly cash flow projections"
                ],
                "benefits": ["Early problem detection", "Better decision making", "Improved planning"]
            }
        }
    
    def _build_tax_knowledge(self) -> Dict:
        """Tanzania-specific tax regulations and guidelines."""
        return {
            "vat": {
                "name": "Value Added Tax (VAT)",
                "rate": "18%",
                "applicability": "Applies to most goods and services",
                "registration_threshold": "100 million TZS annual turnover",
                "filing": "Monthly filing required for registered businesses",
                "exemptions": [
                    "Exports",
                    "Basic food items",
                    "Educational materials",
                    "Medical supplies"
                ],
                "penalties": "2% per month on late payments"
            },
            "withholding_tax": {
                "name": "Withholding Tax",
                "rates": {
                    "services": "5%",
                    "consultancy": "15%",
                    "rent": "10%",
                    "goods": "3%"
                },
                "description": "Tax deducted at source before payment",
                "filing": "Monthly filing with TRA"
            },
            "corporate_tax": {
                "name": "Corporate Income Tax",
                "rate": "30%",
                "applicability": "All resident companies",
                "filing": "Annual returns due 6 months after year-end",
                "estimated_payments": "Quarterly estimated tax payments required"
            },
            "sdl": {
                "name": "Skills Development Levy",
                "rate": "5% of gross salaries",
                "applicability": "All employers",
                "filing": "Monthly with NSSF contributions"
            }
        }
    
    def _build_inventory_strategies(self) -> Dict:
        """Inventory management strategies."""
        return {
            "abc_analysis": {
                "title": "ABC Inventory Analysis",
                "description": "Categorize inventory by value and importance",
                "method": [
                    "A items: High value, 20% of items, 80% of value - strict control",
                    "B items: Medium value, 30% of items, 15% of value - moderate control",
                    "C items: Low value, 50% of items, 5% of value - simple control"
                ],
                "benefits": ["Focus on high-value items", "Optimize stock management"]
            },
            "just_in_time": {
                "title": "Just-In-Time (JIT) Inventory",
                "description": "Order inventory only when needed",
                "pros": ["Lower carrying costs", "Reduced waste", "Better cash flow"],
                "cons": ["Risk of stockouts", "Requires reliable suppliers"],
                "best_for": "Products with predictable demand and reliable suppliers"
            },
            "economic_order_quantity": {
                "title": "Economic Order Quantity (EOQ)",
                "description": "Calculate optimal order size to minimize total inventory costs",
                "formula": "√((2 × Annual Demand × Order Cost) / Holding Cost)",
                "benefits": ["Minimize total costs", "Optimize order frequency"]
            }
        }
    
    def _build_sales_optimization(self) -> Dict:
        """Sales optimization techniques."""
        return {
            "pricing_strategies": {
                "title": "Effective Pricing Strategies",
                "strategies": [
                    "Cost-plus: Add fixed margin to cost",
                    "Value-based: Price based on customer perception",
                    "Competitive: Match or beat competitor prices",
                    "Dynamic: Adjust prices based on demand"
                ],
                "tips": [
                    "Know your costs thoroughly",
                    "Understand your target market",
                    "Monitor competitor pricing",
                    "Test price changes carefully"
                ]
            },
            "upselling": {
                "title": "Upselling & Cross-selling",
                "description": "Increase average transaction value",
                "techniques": [
                    "Bundle related products",
                    "Suggest premium versions",
                    "Offer volume discounts",
                    "Create product sets"
                ],
                "benefits": ["Higher average order value", "Better customer satisfaction"]
            }
        }
    
    def _build_financial_planning(self) -> Dict:
        """Financial planning guidelines."""
        return {
            "budgeting": {
                "title": "Creating Effective Budgets",
                "steps": [
                    "Review historical data",
                    "Set realistic targets",
                    "Plan for seasonal variations",
                    "Include contingency funds",
                    "Review and adjust monthly"
                ],
                "key_budgets": ["Sales budget", "Expense budget", "Cash flow budget"]
            },
            "cash_flow_management": {
                "title": "Cash Flow Management",
                "importance": "Cash is king - profitability doesn't guarantee survival",
                "strategies": [
                    "Maintain cash reserve (3-6 months expenses)",
                    "Negotiate favorable payment terms with suppliers",
                    "Incentivize early customer payments",
                    "Monitor cash flow weekly"
                ]
            }
        }
