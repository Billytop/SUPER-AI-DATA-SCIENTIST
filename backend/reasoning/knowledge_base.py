"""
SEPHLIGHTY AI - KNOWLEDGE BASE & TRAINING INTEGRATION
672 Enterprise Questions - Structured Knowledge System
"""

class KnowledgeBase:
    """
    Contains structured knowledge from 672 training questions.
    Organized by domain for fast lookup and confidence scoring.
    """
    
    # Section A: AI Identity & Limits (Q1-Q42)
    AI_IDENTITY = {
        'what_i_am': [
            "Business Intelligence Brain",
            "Auditor, Accountant, Tax Advisor, Data Scientist",
            "Laravel ERP/POS system intelligence layer",
            "Rule-based + ML hybrid system"
        ],
        'what_i_am_not': [
            "Not a replacement for professional accountant",
            "Not legally binding decision maker",
            "Not authorized to modify database directly",
            "Not a licensed tax consultant"
        ],
        'ethical_boundaries': [
            "Refuse tax evasion suggestions",
            "Refuse to hide transactions",
            "Refuse to bypass audit trails",
            "Refuse unethical requests"
        ],
        'when_to_refuse': [
            "Illegal queries",
            "Unethical manipulation requests",
            "Requests outside trained scope",
            "Insufficient data for accurate answer"
        ]
    }
    
    # Section B: System Discovery (Q43-Q106)
    LARAVEL_AWARENESS = {
        'folders': {
            'app': 'Models, Controllers, Business Logic',
            'routes': 'web.php, api.php - URL routing',
            'database': 'Migrations, Seeders, Schema',
            'config': 'System configuration files',
            'resources': 'Views, Blade templates'
        },
        'modules': [
            'Sales', 'Purchases', 'Inventory', 'Accounting',
            'CRM', 'Reports', 'User Management', 'Settings'
        ],
        'relationships': {
            'belongsTo': 'Many-to-One',
            'hasMany': 'One-to-Many',
            'belongsToMany': 'Many-to-Many with pivot'
        }
    }
    
    # Section C: ERP Core Logic (Q107-Q212)
    ERP_LOGIC = {
        'sales_lifecycle': [
            'Draft → Quotation → Confirmed → Invoiced → Paid',
            'Returns: RMA → Credit Note → Refund/Exchange'
        ],
        'inventory_methods': {
            'FIFO': 'First In First Out - oldest stock sold first',
            'LIFO': 'Last In First Out - newest stock sold first',
            'WAC': 'Weighted Average Cost',
            'Specific': 'Specific identification per item'
        },
        'stock_tracking': [
            'Batch/Lot numbers',
            'Serial numbers',
            'Expiry dates',
            'Multi-location'
        ]
    }
    
    # Section D: Accounting & Audit (Q213-Q336)
    ACCOUNTING = {
        'financial_statements': {
            'P&L': 'Profit & Loss - Revenue minus Expenses',
            'Balance Sheet': 'Assets = Liabilities + Equity',
            'Cash Flow': 'Operating + Investing + Financing',
            'Trial Balance': 'Debit = Credit verification'
        },
        'double_entry': {
            'debit_increases': ['Assets', 'Expenses'],
            'credit_increases': ['Liabilities', 'Equity', 'Revenue']
        },
        'fraud_indicators': [
            'Duplicate transactions',
            'After-hours modifications',
            'Excessive discounts',
            'Round-off anomalies',
            'Ghost employees'
        ]
    }
    
    # Section E: Tax & Country Risk (Q337-Q418)
    TAX_KNOWLEDGE = {
        'tanzania_vat': {
            'standard_rate': '18%',
            'zero_rated': 'Exports, specific essentials',
            'exempt': 'Education, healthcare services'
        },
        'compliance': {
            'TRA': 'Tanzania Revenue Authority',
            'TIN': 'Taxpayer Identification Number',
            'EFD': 'Electronic Fiscal Device required',
            'filing': 'Monthly/Quarterly VAT returns'
        },
        'penalties': [
            'Late filing penalties',
            'Underreporting fines',
            'Interest on overdue payments'
        ],
        'disclaimers': [
            'AI advice not legally binding',
            'Consult professional for complex issues',
            'Tax laws subject to change',
            'User responsibility to verify'
        ]
    }
    
    # Section F: Customer Intelligence (Q419-Q490)
    CUSTOMER_ANALYTICS = {
        'metrics': {
            'CLV': 'Customer Lifetime Value = Avg Order × Frequency × Lifespan',
            'Churn Rate': 'Lost Customers / Total Customers',
            'NPS': 'Net Promoter Score (satisfaction)',
            'Retention Rate': 'Retained / Total at start'
        },
        'aging_buckets': ['0-30 days', '31-60 days', '61-90 days', '90+ days'],
        'risk_indicators': [
            'Declining purchase frequency',
            'Increasing payment delays',
            'Reduced order values',
            'Unresolved complaints'
        ]
    }
    
    # Section G: Employee & HR (Q491-Q552)
    HR_ANALYTICS = {
        'productivity_metrics': [
            'Sales per hour',
            'Transactions per shift',
            'Conversion rate',
            'Average deal size'
        ],
        'misconduct_indicators': [
            'Excessive voids',
            'Unusual discount patterns',
            'After-hours access',
            'Permission abuse'
        ],
        'fairness_principles': [
            'Objective metrics over subjective',
            'Experience-adjusted comparisons',
            'Role-based standards',
            'Bias detection'
        ]
    }
    
    # Section H: AI Analytics (Q553-Q634)
    DATA_SCIENCE = {
        'forecasting': {
            'methods': ['Moving Average', 'Exponential Smoothing', 'ARIMA', 'Prophet'],
            'accuracy': 'MAPE (Mean Absolute Percentage Error)',
            'confidence': 'Prediction intervals provided'
        },
        'correlation_vs_causation': [
            'Correlation ≠ Causation',
            'Need temporal precedence',
            'Control for confounding variables',
            'A/B testing for proof'
        ],
        'explainability': [
            'Feature importance ranking',
            'SHAP values',
            'Business-friendly language',
            'Transparent reasoning'
        ]
    }
    
    # Section I: Security & Health (Q635-Q672)
    SECURITY = {
        'threats': [
            'Unauthorized access attempts',
            'Data tampering',
            'Permission abuse',
            'Credential stuffing',
            'Session hijacking'
        ],
        'monitoring': [
            'Failed login attempts',
            'Privilege escalation',
            'After-hours activity',
            'Bulk modifications'
        ]
    }
    
    # Section J: Conversation Intelligence (Q673-Q732)
    CONVERSATION = {
        'date_awareness': {
            'today': 'Current date',
            'this_week': 'Monday to Sunday current week',
            'this_month': 'First to last day of current month',
            'this_year': 'Jan 1 to Dec 31 current year'
        },
        'refusal_templates': {
            'out_of_scope': "I don't have training on that topic. I can help with: {suggestions}",
            'unethical': "I cannot assist with that request as it violates ethical guidelines.",
            'insufficient_data': "I need more information: {missing_fields}",
            'low_confidence': "I'm not confident about this (confidence: {score}%). Let me clarify: {question}"
        }
    }
    
    @staticmethod
    def get_confidence_threshold(domain):
        """
        Return confidence thresholds by domain.
        Financial/Tax queries require higher confidence.
        """
        thresholds = {
            'financial': 90,
            'tax': 90,
            'accounting': 85,
            'sales': 75,
            'inventory': 75,
            'customer': 70,
            'general': 60
        }
        return thresholds.get(domain, 70)
    
    @staticmethod
    def get_domain_from_intent(intent):
        """Map intent to knowledge domain"""
        domain_map = {
            'SALES': 'sales',
            'CUSTOMER_RISK': 'customer',
            'INVENTORY': 'inventory',
            'EMPLOYEE_PERF': 'general',
            'TAX': 'tax',
            'AUDIT': 'accounting',
            'FORECAST': 'financial'
        }
        return domain_map.get(intent, 'general')
    
    @staticmethod
    def get_smart_suggestions(query_lower, lang='en'):
        """
        Generate context-aware suggestions based on query keywords.
        Returns (suggestions_list, confidence_score)
        """
        suggestions = []
        confidence = 30  # Default low
        
        # Financial/Accounting
        if any(x in query_lower for x in ['balance', 'sheet', 'ledger', 'p&l', 'profit', 'loss']):
            suggestions = [
                "Profit ya mwaka" if lang == 'sw' else "Annual profit",
                "Balance sheet" if lang == 'en' else "Balancesheet",
                "Cash flow statement"
            ]
            confidence = 50
        
        # Tax queries
        elif any(x in query_lower for x in ['vat', 'tax', 'tra', 'compliance']):
            suggestions = [
                "VAT payable",
                "Tax compliance status",
                "Filing deadline"
            ]
            confidence = 55
        
        # Sales
        elif any(x in query_lower for x in ['sale', 'sell', 'mauzo', 'revenue']):
            suggestions = [
                "Mauzo ya leo" if lang == 'sw' else "Sales today",
                "Mauzo ya mwezi" if lang == 'sw' else "Monthly sales",
                "Bidhaa bora" if lang == 'sw' else "Best products"
            ]
            confidence = 60
        
        # Inventory
        elif any(x in query_lower for x in ['stock', 'inventory', 'hisa', 'bidhaa']):
            suggestions = [
                "Stock movement",
                "Bidhaa zilizo chini" if lang == 'sw' else "Low stock items",
                "Inventory valuation"
            ]
            confidence = 58
        
        # Customer/Debt
        elif any(x in query_lower for x in ['customer', 'debt', 'deni', 'mteja']):
            suggestions = [
                "Deni la [jina]" if lang == 'sw' else "Debt of [name]",
                "Customer aging report",
                "High risk customers"
            ]
            confidence = 55
        
        # Employee
        elif any(x in query_lower for x in ['employee', 'staff', 'mfanyakazi', 'performance']):
            suggestions = [
                "Best employee",
                "Employee performance",
                "Productivity comparison"
            ]
            confidence = 52
        
        # Forecasting
        elif any(x in query_lower for x in ['forecast', 'predict', 'trend', 'future']):
            suggestions = [
                "Sales forecast",
                "Demand prediction",
                "Trend analysis"
            ]
            confidence = 48
        
        return suggestions, confidence
