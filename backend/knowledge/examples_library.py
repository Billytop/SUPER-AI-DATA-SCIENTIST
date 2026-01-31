"""
Examples Library
Query examples, use cases, and tutorials for common tasks.
"""

from typing import Dict, List


class ExamplesLibrary:
    """
    Library of example queries, expected outputs, and use cases.
    """
    
    def __init__(self):
        self.query_examples = self._build_query_examples()
        self.use_cases = self._build_use_cases()
        self.tutorials = self._build_tutorials()
        
    def get_examples_for_intent(self, intent: str) -> List[Dict]:
        """Get example queries for an intent."""
        return self.query_examples.get(intent, [])
    
    def get_use_case(self, scenario: str) -> Optional[Dict]:
        """Get use case example."""
        for key, case in self.use_cases.items():
            if scenario.lower() in key:
                return case
        return None
    
    def get_tutorial(self, topic: str) -> Optional[Dict]:
        """Get tutorial for a topic."""
        for key, tutorial in self.tutorials.items():
            if topic.lower() in key:
                return tutorial
        return None
    
    def _build_query_examples(self) -> Dict:
        """Build example queries per intent."""
        return {
            "SALES": [
                {
                    "query": "sales today",
                    "expected_intent": "SALES",
                    "expected_output": "Daily sales total with transaction count"
                },
                {
                    "query": "mauzo wiki hii",
                    "expected_intent": "SALES",
                    "expected_output": "Weekly sales breakdown"
                },
                {
                    "query": "compare sales this month vs last month",
                    "expected_intent": "COMPARISON",
                    "expected_output": "Monthly comparison with percentage change"
                }
            ],
            "INVENTORY": [
                {
                    "query": "stock value",
                    "expected_intent": "INVENTORY",
                    "expected_output": "Total inventory valuation"
                },
                {
                    "query": "low stock items",
                    "expected_intent": "INVENTORY",
                    "expected_output": "Products below reorder point"
                }
            ],
            "CUSTOMER_RISK": [
                {
                    "query": "customer with debt",
                    "expected_intent": "CUSTOMER_RISK",
                    "expected_output": "List of customers with outstanding balances"
                },
                {
                    "query": "debt aging",
                    "expected_intent": "CUSTOMER_RISK",
                    "expected_output": "Aging analysis of receivables"
                }
            ],
            "BEST_PRODUCT": [
                {
                    "query": "best products",
                    "expected_intent": "BEST_PRODUCT",
                    "expected_output": "Top 5 products by quantity sold"
                },
                {
                    "query": "best products by revenue",
                    "expected_intent": "BEST_PRODUCT",
                    "expected_output": "Top 5 products by sales value"
                }
            ]
        }
    
    def _build_use_cases(self) -> Dict:
        """Build common use case scenarios."""
        return {
            "daily_operations": {
                "title": "Daily Operations Checklist",
                "scenario": "Start of day routine for shop manager",
                "queries": [
                    "sales yesterday",
                    "low stock items",
                    "customer payments today",
                    "best selling products this week"
                ],
                "purpose": "Quick morning overview of business status"
            },
            "month_end": {
                "title": "Month-End Review",
                "scenario": "Monthly performance analysis",
                "queries": [
                    "sales this month",
                    "compare sales this month vs last month",
                    "profit this month",
                    "debt aging",
                    "top employees this month"
                ],
                "purpose": "Comprehensive monthly business review"
            },
            "inventory_audit": {
                "title": "Inventory Audit",
                "scenario": "Regular inventory review and optimization",
                "queries": [
                    "stock value",
                    "low stock items",
                    "slow moving products",
                    "stock turnover rate"
                ],
                "purpose": "Optimize inventory levels and identify issues"
            }
        }
    
    def _build_tutorials(self) -> Dict:
        """Build step-by-step tutorials."""
        return {
            "debt_collection": {
                "title": "How to Manage Customer Debt",
                "steps": [
                    "1. Run 'customer with debt' query to see all outstanding balances",
                    "2. Check 'debt aging' to prioritize oldest debts",
                    "3. Contact customers with overdue payments",
                    "4. Set credit limits for repeat late payers",
                    "5. Monitor weekly to prevent accumulation"
                ],
                "best_practices": [
                    "Follow up within 7 days of invoice due date",
                    "Be consistent with payment terms",
                    "Reward early payers with discounts"
                ]
            },
            "sales_analysis": {
                "title": "How to Analyze Sales Performance",
                "steps": [
                    "1. Check 'sales today' for current performance",
                    "2. Compare 'sales this week vs last week'",
                    "3. Identify 'best products' to focus on winners",
                    "4. Review 'sales by category' for trends",
                    "5. Analyze 'sales by employee' for team performance"
                ],
                "metrics_to_track": [
                    "Total revenue",
                    "Average order value",
                    "Number of transactions",
                    "Best performing products/categories"
                ]
            }
        }
