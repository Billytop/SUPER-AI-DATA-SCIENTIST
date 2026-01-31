"""
Business Rules Engine
Automated business logic execution with configurable rules, conditions, and actions.
"""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import re


class RuleConditionOperator(Enum):
    """Operators for rule conditions."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    IN_LIST = "in"
    MATCHES_PATTERN = "matches"
    BETWEEN = "between"


class RuleAction(Enum):
    """Types of actions rules can trigger."""
    SEND_NOTIFICATION = "send_notification"
    UPDATE_FIELD = "update_field"
    CALCULATE_VALUE = "calculate_value"
    TRIGGER_WORKFLOW = "trigger_workflow"
    APPLY_DISCOUNT = "apply_discount"
    BLOCK_TRANSACTION = "block_transaction"
    CREATE_TASK = "create_task"
    LOG_EVENT = "log_event"


class BusinessRulesEngine:
    """
    Executes business rules with complex conditions and actions.
    """
    
    def __init__(self):
        self.rules = {}
        self.rule_sets = {}
        self.execution_log = []
        self.rule_counter = 0
        
    def create_rule(self, name: str, description: str, conditions: List[Dict], actions: List[Dict], priority: int = 100) -> str:
        """
        Create a business rule.
        
        Args:
            name: Rule name
            description: Rule description
            conditions: List of conditions (AND logic)
            actions: List of actions to execute
            priority: Execution priority (higher = first)
            
        Returns:
            Rule ID
        """
        self.rule_counter += 1
        rule_id = f"rule_{self.rule_counter}"
        
        self.rules[rule_id] = {
            'id': rule_id,
            'name': name,
            'description': description,
            'conditions': conditions,
            'actions': actions,
            'priority': priority,
            'enabled': True,
            'execution_count': 0,
            'created_at': datetime.now().isoformat()
        }
        
        return rule_id
    
    def execute_rules(self, context: Dict, rule_set: str = None) -> Dict:
        """
        Execute rules against context data.
        
        Args:
            context: Data context to evaluate
            rule_set: Optional specific rule set to execute
            
        Returns:
            Execution results
        """
        # Get applicable rules
        if rule_set and rule_set in self.rule_sets:
            rule_ids = self.rule_sets[rule_set]
            applicable_rules = [self.rules[rid] for rid in rule_ids if rid in self.rules]
        else:
            applicable_rules = list(self.rules.values())
        
        # Filter enabled rules
        applicable_rules = [r for r in applicable_rules if r['enabled']]
        
        # Sort by priority
        applicable_rules.sort(key=lambda r: r['priority'], reverse=True)
        
        # Execute rules
        executed_actions = []
        triggered_rules = []
        
        for rule in applicable_rules:
            if self._evaluate_conditions(rule['conditions'], context):
                # Execute actions
                for action in rule['actions']:
                    result = self._execute_action(action, context)
                    executed_actions.append(result)
                
                triggered_rules.append(rule['id'])
                rule['execution_count'] += 1
                
                # Log execution
                self._log_execution(rule['id'], context, executed_actions)
        
        return {
            'triggered_rules': triggered_rules,
            'executed_actions': executed_actions,
            'evaluated_at': datetime.now().isoformat()
        }
    
    def create_pricing_rules(self) -> List[str]:
        """Create common pricing business rules."""
        rules = []
        
        # Volume discount rule
        rules.append(self.create_rule(
            'Volume Discount - 10+ Items',
            'Apply 10% discount for orders with 10+ items',
            [
                {'field': 'quantity', 'operator': '>=', 'value': 10}
            ],
            [
                {'type': 'apply_discount', 'params': {'percent': 10, 'reason': 'Volume discount'}}
            ],
            priority=100
        ))
        
        # Bulk order discount
        rules.append(self.create_rule(
            'Bulk Order - 1M+ TZS',
            'Apply 5% discount for orders over 1M TZS',
            [
                {'field': 'order_total', 'operator': '>', 'value': 1000000}
            ],
            [
                {'type': 'apply_discount', 'params': {'percent': 5, 'reason': 'Bulk order discount'}}
            ],
            priority=90
        ))
        
        # VIP customer pricing
        rules.append(self.create_rule(
            'VIP Customer Discount',
            'Apply 15% discount for VIP customers',
            [
                {'field': 'customer_segment', 'operator': '==', 'value': 'VIP'}
            ],
            [
                {'type': 'apply_discount', 'params': {'percent': 15, 'reason': 'VIP customer discount'}}
            ],
            priority=120
        ))
        
        return rules
    
    def create_credit_rules(self) -> List[str]:
        """Create credit and payment business rules."""
        rules = []
        
        # Block sale if credit limit exceeded
        rules.append(self.create_rule(
            'Credit Limit Check',
            'Block transaction if customer exceeds credit limit',
            [
                {'field': 'customer_debt', 'operator': '>', 'value': '${customer_credit_limit}'}
            ],
            [
                {'type': 'block_transaction', 'params': {'reason': 'Credit limit exceeded'}},
                {'type': 'send_notification', 'params': {
                    'to': 'manager',
                    'message': 'Customer ${customer_name} credit limit exceeded'
                }}
            ],
            priority=150
        ))
        
        # Require approval for large credit sales
        rules.append(self.create_rule(
            'Large Credit Sale Approval',
            'Require manager approval for credit sales over 5M TZS',
            [
                {'field': 'payment_method', 'operator': '==', 'value': 'credit'},
                {'field': 'order_total', 'operator': '>', 'value': 5000000}
            ],
            [
                {'type': 'create_task', 'params': {
                    'task_type': 'approval',
                    'assignee': 'manager',
                    'description': 'Approve credit sale of ${order_total} TZS'
                }}
            ],
            priority=140
        ))
        
        # Payment reminder
        rules.append(self.create_rule(
            'Payment Overdue Reminder',
            'Send reminder for overdue payments',
            [
                {'field': 'payment_due_date', 'operator': '<', 'value': 'today'},
                {'field': 'payment_status', 'operator': '==', 'value': 'pending'}
            ],
            [
                {'type': 'send_notification', 'params': {
                    'to': 'customer',
                    'template': 'payment_reminder'
                }}
            ],
            priority=100
        ))
        
        return rules
    
    def create_inventory_rules(self) -> List[str]:
        """Create inventory management business rules."""
        rules = []
        
        # Auto-reorder low stock
        rules.append(self.create_rule(
            'Auto Reorder Low Stock',
            'Create purchase order when stock falls below reorder point',
            [
                {'field': 'stock_quantity', 'operator': '<=', 'value': '${reorder_point}'},
                {'field': 'auto_reorder_enabled', 'operator': '==', 'value': True}
            ],
            [
                {'type': 'trigger_workflow', 'params': {
                    'workflow': 'create_purchase_order',
                    'quantity': '${reorder_quantity}'
                }},
                {'type': 'send_notification', 'params': {
                    'to': 'purchasing',
                    'message': 'Auto-reorder triggered for ${product_name}'
                }}
            ],
            priority=110
        ))
        
        # Block sale for out of stock
        rules.append(self.create_rule(
            'Prevent Overselling',
            'Block sale if insufficient stock',
            [
                {'field': 'requested_quantity', 'operator': '>', 'value': '${available_stock}'}
            ],
            [
                {'type': 'block_transaction', 'params': {
                    'reason': 'Insufficient stock available'
                }}
            ],
            priority=160
        ))
        
        # Mark slow-moving items
        rules.append(self.create_rule(
            'Identify Slow-Moving Stock',
            'Flag items with low sales velocity',
            [
                {'field': 'days_since_last_sale', 'operator': '>', 'value': 90},
                {'field': 'stock_quantity', 'operator': '>', 'value': 10}
            ],
            [
                {'type': 'update_field', 'params': {
                    'field': 'slow_moving',
                    'value': True
                }},
                {'type': 'create_task', 'params': {
                    'task_type': 'review',
                    'assignee': 'inventory_manager',
                    'description': 'Review slow-moving item: ${product_name}'
                }}
            ],
            priority=80
        ))
        
        return rules
    
    def create_sales_validation_rules(self) -> List[str]:
        """Create sales transaction validation rules."""
        rules = []
        
        # Validate transaction total
        rules.append(self.create_rule(
            'Validate Transaction Total',
            'Ensure transaction total matches line items',
            [
                {'field': 'calculated_total', 'operator': '!=', 'value': '${transaction_total}'}
            ],
            [
                {'type': 'block_transaction', 'params': {
                    'reason': 'Transaction total mismatch'
                }},
                {'type': 'log_event', 'params': {
                    'level': 'error',
                    'message': 'Total mismatch: calculated=${calculated_total}, actual=${transaction_total}'
                }}
            ],
            priority=170
        ))
        
        # Validate customer exists
        rules.append(self.create_rule(
            'Validate Customer',
            'Ensure customer exists for transaction',
            [
                {'field': 'customer_id', 'operator': '!=', 'value': None},
                {'field': 'customer_exists', 'operator': '==', 'value': False}
            ],
            [
                {'type': 'block_transaction', 'params': {
                    'reason': 'Invalid customer ID'
                }}
            ],
            priority=180
        ))
        
        # Require manager approval for large discounts
        rules.append(self.create_rule(
            'Manager Approval - Large Discount',
            'Require approval for discounts over 20%',
            [
                {'field': 'discount_percent', 'operator': '>', 'value': 20}
            ],
            [
                {'type': 'create_task', 'params': {
                    'task_type': 'approval',
                    'assignee': 'manager',
                    'priority': 'high',
                    'description': 'Approve ${discount_percent}% discount on order ${order_id}'
                }}
            ],
            priority=130
        ))
        
        return rules
    
    def create_customer_management_rules(self) -> List[str]:
        """Create customer relationship management rules."""
        rules = []
        
        # Upgrade to VIP
        rules.append(self.create_rule(
            'Auto Upgrade to VIP',
            'Automatically upgrade customer to VIP segment',
            [
                {'field': 'lifetime_value', 'operator': '>', 'value': 10000000},
                {'field': 'customer_segment', 'operator': '!=', 'value': 'VIP'}
            ],
            [
                {'type': 'update_field', 'params': {
                    'field': 'customer_segment',
                    'value': 'VIP'
                }},
                {'type': 'send_notification', 'params': {
                    'to': 'customer',
                    'template': 'vip_upgrade'
                }},
                {'type': 'send_notification', 'params': {
                    'to': 'sales_team',
                    'message': 'Customer ${customer_name} upgraded to VIP'
                }}
            ],
            priority=100
        ))
        
        # Flag inactive customers
        rules.append(self.create_rule(
            'Flag Inactive Customers',
            'Mark customers with no purchases in 6 months',
            [
                {'field': 'days_since_last_purchase', 'operator': '>', 'value': 180}
            ],
            [
                {'type': 'update_field', 'params': {
                    'field': 'status',
                    'value': 'inactive'
                }},
                {'type': 'create_task', 'params': {
                    'task_type': 'follow_up',
                    'assignee': 'sales_team',
                    'description': 'Re-engagement campaign for ${customer_name}'
                }}
            ],
            priority=70
        ))
        
        return rules
    
    def create_tax_calculation_rules(self) -> List[str]:
        """Create tax calculation business rules."""
        rules = []
        
        # Apply VAT
        rules.append(self.create_rule(
            'Apply VAT - 18%',
            'Calculate and apply 18% VAT on taxable items',
            [
                {'field': 'product_taxable', 'operator': '==', 'value': True}
            ],
            [
                {'type': 'calculate_value', 'params': {
                    'field': 'vat_amount',
                    'formula': 'subtotal * 0.18'
                }},
                {'type': 'calculate_value', 'params': {
                    'field': 'total_with_vat',
                    'formula': 'subtotal * 1.18'
                }}
            ],
            priority=110
        ))
        
        # Export exemption
        rules.append(self.create_rule(
            'VAT Exemption for Exports',
            'Zero-rate VAT for export transactions',
            [
                {'field': 'transaction_type', 'operator': '==', 'value': 'export'}
            ],
            [
                {'type': 'calculate_value', 'params': {
                    'field': 'vat_rate',
                    'formula': '0'
                }},
                {'type': 'update_field', 'params': {
                    'field': 'vat_exemption_reason',
                    'value': 'Export transaction'
                }}
            ],
            priority=120
        ))
        
        return rules
    
    def _evaluate_conditions(self, conditions: List[Dict], context: Dict) -> bool:
        """Evaluate if all conditions are met (AND logic)."""
        for condition in conditions:
            if not self._evaluate_single_condition(condition, context):
                return False
        return True
    
    def _evaluate_single_condition(self, condition: Dict, context: Dict) -> bool:
        """Evaluate a single condition."""
        field = condition['field']
        operator = condition['operator']
        expected_value = condition['value']
        
        # Resolve field value from context
        actual_value = self._resolve_value(field, context)
        
        # Resolve expected value if it's a context variable
        if isinstance(expected_value, str) and expected_value.startswith('${') and expected_value.endswith('}'):
            var_name = expected_value[2:-1]
            expected_value = context.get(var_name)
        
        # Evaluate based on operator
        if operator == '==':
            return actual_value == expected_value
        elif operator == '!=':
            return actual_value != expected_value
        elif operator == '>':
            return actual_value > expected_value
        elif operator == '<':
            return actual_value < expected_value
        elif operator == '>=':
            return actual_value >= expected_value
        elif operator == '<=':
            return actual_value <= expected_value
        elif operator == 'contains':
            return expected_value in str(actual_value)
        elif operator == 'in':
            return actual_value in expected_value
        elif operator == 'matches':
            return bool(re.match(expected_value, str(actual_value)))
        elif operator == 'between':
            return expected_value[0] <= actual_value <= expected_value[1]
        else:
            return False
    
    def _execute_action(self, action: Dict, context: Dict) -> Dict:
        """Execute a rule action."""
        action_type = action['type']
        params = action.get('params', {})
        
        # Resolve parameter values from context
        resolved_params = {}
        for key, value in params.items():
            resolved_params[key] = self._resolve_value(value, context)
        
        result = {
            'action_type': action_type,
            'params': resolved_params,
            'executed_at': datetime.now().isoformat()
        }
        
        # Simulate action execution (in production, would trigger actual actions)
        if action_type == 'send_notification':
            result['status'] = 'notification_sent'
        elif action_type == 'update_field':
            result['status'] = 'field_updated'
        elif action_type == 'block_transaction':
            result['status'] = 'transaction_blocked'
        elif action_type == 'apply_discount':
            result['status'] = 'discount_applied'
        elif action_type == 'calculate_value':
            result['status'] = 'value_calculated'
        else:
            result['status'] = 'executed'
        
        return result
    
    def _resolve_value(self, value: Any, context: Dict) -> Any:
        """Resolve value from context if it's a variable reference."""
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            var_name = value[2:-1]
            return context.get(var_name, value)
        
        # Special keywords
        if value == 'today':
            return datetime.now().date().isoformat()
        elif value == 'now':
            return datetime.now().isoformat()
        
        return value
    
    def _log_execution(self, rule_id: str, context: Dict, actions: List[Dict]):
        """Log rule execution."""
        self.execution_log.append({
            'rule_id': rule_id,
            'executed_at': datetime.now().isoformat(),
            'context_keys': list(context.keys()),
            'actions_executed': len(actions)
        })
        
        # Keep only last 1000 executions
        self.execution_log = self.execution_log[-1000:]
    
    def get_rule_statistics(self) -> Dict:
        """Get rule execution statistics."""
        total_rules = len(self.rules)
        enabled_rules = sum(1 for r in self.rules.values() if r['enabled'])
        
        # Most executed rules
        sorted_by_execution = sorted(
            self.rules.values(),
            key=lambda r: r['execution_count'],
            reverse=True
        )[:10]
        
        return {
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'disabled_rules': total_rules - enabled_rules,
            'total_executions': len(self.execution_log),
            'most_executed_rules': [
                {'id': r['id'], 'name': r['name'], 'executions': r['execution_count']}
                for r in sorted_by_execution
            ]
        }
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id]['enabled'] = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id]['enabled'] = False
            return True
        return False
    
    def create_rule_set(self, name: str, rule_ids: List[str]) -> str:
        """Group rules into a rule set."""
        self.rule_sets[name] = rule_ids
        return name
    
    def get_applicable_rules(self, context: Dict) -> List[Dict]:
        """Get rules that would be triggered by given context."""
        applicable = []
        
        for rule in self.rules.values():
            if not rule['enabled']:
                continue
            
            if self._evaluate_conditions(rule['conditions'], context):
                applicable.append({
                    'id': rule['id'],
                    'name': rule['name'],
                    'priority': rule['priority'],
                    'actions': rule['actions']
                })
        
        return sorted(applicable, key=lambda r: r['priority'], reverse=True)


class RuleTemplate:
    """
    Pre-built rule templates for common scenarios.
    """
    
    @staticmethod
    def minimum_order_value(min_value: float) -> Dict:
        """Template for minimum order value rule."""
        return {
            'name': f'Minimum Order Value - {min_value:,.0f} TZS',
            'description': f'Block orders below {min_value:,.0f} TZS',
            'conditions': [
                {'field': 'order_total', 'operator': '<', 'value': min_value}
            ],
            'actions': [
                {'type': 'block_transaction', 'params': {
                    'reason': f'Order total below minimum of {min_value:,.0f} TZS'
                }}
            ],
            'priority': 150
        }
    
    @staticmethod
    def loyalty_points(points_per_amount: int, amount_threshold: float) -> Dict:
        """Template for loyalty points calculation."""
        return {
            'name': 'Calculate Loyalty Points',
            'description': f'Award {points_per_amount} points per {amount_threshold:,.0f} TZS',
            'conditions': [
                {'field': 'order_total', 'operator': '>=', 'value': amount_threshold}
            ],
            'actions': [
                {'type': 'calculate_value', 'params': {
                    'field': 'loyalty_points_earned',
                    'formula': f'floor(order_total / {amount_threshold}) * {points_per_amount}'
                }}
            ],
            'priority': 90
        }
    
    @staticmethod
    def seasonal_discount(season: str, discount_percent: float, start_date: str, end_date: str) -> Dict:
        """Template for seasonal discount."""
        return {
            'name': f'{season} Seasonal Discount',
            'description': f'{discount_percent}% discount during {season}',
            'conditions': [
                {'field': 'current_date', 'operator': 'between', 'value': [start_date, end_date]}
            ],
            'actions': [
                {'type': 'apply_discount', 'params': {
                    'percent': discount_percent,
                    'reason': f'{season} seasonal discount'
                }}
            ],
            'priority': 85
        }
