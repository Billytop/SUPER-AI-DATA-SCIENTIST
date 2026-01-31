"""
Smart Notification System
Alert engine with conditions, multi-channel delivery (SMS/Email), and preferences.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


class AlertEngine:
    """
    Creates intelligent alerts based on business conditions.
    """
    
    def __init__(self):
        self.alert_rules = {}
        self.triggered_alerts = []
        self.rule_counter = 0
        
    def create_alert_rule(self, name: str, condition: Dict, action: Dict, priority: str = 'medium') -> str:
        """
        Create alert rule.
        
        Args:
            name: Rule name
            condition: Alert condition (metric, operator, threshold)
            action: Action to take (notification, webhook, etc.)
            priority: Alert priority
            
        Returns:
            Rule ID
        """
        self.rule_counter += 1
        rule_id = f"rule_{self.rule_counter}"
        
        self.alert_rules[rule_id] = {
            'id': rule_id,
            'name': name,
            'condition': condition,
            'action': action,
            'priority': priority,
            'enabled': True,
            'trigger_count': 0,
            'last_triggered': None,
            'created_at': datetime.now().isoformat()
        }
        
        return rule_id
    
    def check_conditions(self, data: Dict) -> List[Dict]:
        """Check all rules against current data."""
        triggered = []
        
        for rule_id, rule in self.alert_rules.items():
            if not rule['enabled']:
                continue
            
            if self._evaluate_condition(rule['condition'], data):
                alert = self._trigger_alert(rule_id, data)
                triggered.append(alert)
        
        return triggered
    
    def create_inventory_alerts(self) -> List[str]:
        """Create common inventory alert rules."""
        rules = []
        
        # Low stock alert
        rules.append(self.create_alert_rule(
            'Low Stock Alert',
            {'metric': 'stock_quantity', 'operator': '<', 'threshold': 10},
            {'type': 'notification', 'channels': ['email', 'in_app']},
            'high'
        ))
        
        # Out of stock alert
        rules.append(self.create_alert_rule(
            'Out of Stock',
            {'metric': 'stock_quantity', 'operator': '==', 'threshold': 0},
            {'type': 'notification', 'channels': ['email', 'sms', 'in_app']},
            'urgent'
        ))
        
        # Overstocking alert
        rules.append(self.create_alert_rule(
            'Overstock Warning',
            {'metric': 'stock_quantity', 'operator': '>', 'threshold': 1000},
            {'type': 'notification', 'channels': ['email']},
            'medium'
        ))
        
        return rules
    
    def create_financial_alerts(self) -> List[str]:
        """Create financial alert rules."""
        rules = []
        
        # Large expense alert
        rules.append(self.create_alert_rule(
            'Large Expense',
            {'metric': 'expense_amount', 'operator': '>', 'threshold': 1000000},
            {'type': 'notification', 'channels': ['email', 'sms']},
            'high'
        ))
        
        # Revenue drop alert
        rules.append(self.create_alert_rule(
            'Revenue Drop',
            {'metric': 'daily_revenue_change_percent', 'operator': '<', 'threshold': -20},
            {'type': 'notification', 'channels': ['email', 'in_app']},
            'urgent'
        ))
        
        # Debt threshold alert
        rules.append(self.create_alert_rule(
            'Customer Debt Exceeded',
            {'metric': 'customer_debt', 'operator': '>', 'threshold': 5000000},
            {'type': 'notification', 'channels': ['email']},
            'high'
        ))
        
        return rules
    
    def create_sales_alerts(self) -> List[str]:
        """Create sales performance alert rules."""
        rules = []
        
        # Sales spike (opportunity)
        rules.append(self.create_alert_rule(
            'Sales Spike Detected',
            {'metric': 'daily_sales_change_percent', 'operator': '>', 'threshold': 50},
            {'type': 'notification', 'channels': ['email', 'in_app']},
            'medium'
        ))
        
        # Sales slowdown
        rules.append(self.create_alert_rule(
            'Sales Slowdown',
            {'metric': 'daily_sales_change_percent', 'operator': '<', 'threshold': -30},
            {'type': 'notification', 'channels': ['email', 'sms']},
            'high'
        ))
        
        return rules
    
    def _evaluate_condition(self, condition: Dict, data: Dict) -> bool:
        """Evaluate if condition is met."""
        metric = condition['metric']
        operator = condition['operator']
        threshold = condition['threshold']
        
        if metric not in data:
            return False
        
        value = data[metric]
        
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '==':
            return value == threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        else:
            return False
    
    def _trigger_alert(self, rule_id: str, data: Dict) -> Dict:
        """Trigger an alert."""
        rule = self.alert_rules[rule_id]
        
        alert = {
            'id': f"alert_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            'rule_id': rule_id,
            'rule_name': rule['name'],
            'priority': rule['priority'],
            'triggered_at': datetime.now().isoformat(),
            'data': data,
            'action': rule['action']
        }
        
        # Update rule stats
        rule['trigger_count'] += 1
        rule['last_triggered'] = alert['triggered_at']
        
        # Store alert
        self.triggered_alerts.append(alert)
        
        return alert


class NotificationSender:
    """
    Sends notifications through multiple channels.
    """
    
    def __init__(self):
        self.sent_notifications = []
        self.delivery_log = []
        
    def send_email(self, to: str, subject: str, body: str, priority: str = 'medium') -> Dict:
        """Send email notification."""
        notification_id = f"email_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        notification = {
            'id': notification_id,
            'channel': 'email',
            'to': to,
            'subject': subject,
            'body': body,
            'priority': priority,
            'sent_at': datetime.now().isoformat(),
            'status': 'sent'
        }
        
        self.sent_notifications.append(notification)
        self._log_delivery(notification)
        
        return notification
    
    def send_sms(self, phone: str, message: str, priority: str = 'medium') -> Dict:
        """Send SMS notification."""
        notification_id = f"sms_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        notification = {
            'id': notification_id,
            'channel': 'sms',
            'phone': phone,
            'message': message,
            'priority': priority,
            'sent_at': datetime.now().isoformat(),
            'status': 'sent'
        }
        
        self.sent_notifications.append(notification)
        self._log_delivery(notification)
        
        return notification
    
    def send_push(self, user_id: str, title: str, message: str, data: Dict = None) -> Dict:
        """Send push notification."""
        notification_id = f"push_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        notification = {
            'id': notification_id,
            'channel': 'push',
            'user_id': user_id,
            'title': title,
            'message': message,
            'data': data or {},
            'sent_at': datetime.now().isoformat(),
            'status': 'sent'
        }
        
        self.sent_notifications.append(notification)
        self._log_delivery(notification)
        
        return notification
    
    def send_in_app(self, user_id: str, message: str, action_url: str = '') -> Dict:
        """Send in-app notification."""
        notification_id = f"inapp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        notification = {
            'id': notification_id,
            'channel': 'in_app',
            'user_id': user_id,
            'message': message,
            'action_url': action_url,
            'sent_at': datetime.now().isoformat(),
            'read': False,
            'status': 'delivered'
        }
        
        self.sent_notifications.append(notification)
        self._log_delivery(notification)
        
        return notification
    
    def send_multi_channel(self, channels: List[str], recipients: Dict, message_data: Dict) -> List[Dict]:
        """Send notification across multiple channels."""
        results = []
        
        if 'email' in channels and 'email' in recipients:
            result = self.send_email(
                recipients['email'],
                message_data.get('subject', 'Notification'),
                message_data.get('body', '')
            )
            results.append(result)
        
        if 'sms' in channels and 'phone' in recipients:
            result = self.send_sms(
                recipients['phone'],
                message_data.get('message', '')[:160]  # SMS length limit
            )
            results.append(result)
        
        if 'push' in channels and 'user_id' in recipients:
            result = self.send_push(
                recipients['user_id'],
                message_data.get('title', 'Notification'),
                message_data.get('message', '')
            )
            results.append(result)
        
        if 'in_app' in channels and 'user_id' in recipients:
            result = self.send_in_app(
                recipients['user_id'],
                message_data.get('message', '')
            )
            results.append(result)
        
        return results
    
    def get_delivery_stats(self) -> Dict:
        """Get notification delivery statistics."""
        total = len(self.sent_notifications)
        
        if total == 0:
            return {'total_sent': 0}
        
        by_channel = {}
        by_priority = {}
        
        for notif in self.sent_notifications:
            channel = notif['channel']
            by_channel[channel] = by_channel.get(channel, 0) + 1
            
            priority = notif.get('priority', 'medium')
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        return {
            'total_sent': total,
            'by_channel': by_channel,
            'by_priority': by_priority
        }
    
    def _log_delivery(self, notification: Dict):
        """Log notification delivery."""
        self.delivery_log.append({
            'notification_id': notification['id'],
            'channel': notification['channel'],
            'timestamp': notification['sent_at'],
            'status': notification['status']
        })


class NotificationPreferences:
    """
    Manages user notification preferences.
    """
    
    def __init__(self):
        self.user_preferences = {}
        
    def set_preferences(self, user_id: str, preferences: Dict):
        """Set user notification preferences."""
        self.user_preferences[user_id] = {
            'enabled_channels': preferences.get('enabled_channels', ['email', 'in_app']),
            'quiet_hours': preferences.get('quiet_hours', {'start': '22:00', 'end': '07:00'}),
            'alert_types': preferences.get('alert_types', {
                'inventory': True,
                'sales': True,
                'financial': True,
                'system': True
            }),
            'min_priority': preferences.get('min_priority', 'medium'),
            'updated_at': datetime.now().isoformat()
        }
    
    def get_preferences(self, user_id: str) -> Dict:
        """Get user notification preferences."""
        return self.user_preferences.get(user_id, self._default_preferences())
    
    def should_notify(self, user_id: str, notification: Dict) -> bool:
        """Check if user should receive notification based on preferences."""
        prefs = self.get_preferences(user_id)
        
        # Check channel
        if notification['channel'] not in prefs['enabled_channels']:
            return False
        
        # Check quiet hours
        if self._is_quiet_hours(prefs['quiet_hours']):
            if notification.get('priority') != 'urgent':
                return False
        
        # Check priority
        priority_levels = ['low', 'medium', 'high', 'urgent']
        min_priority_index = priority_levels.index(prefs['min_priority'])
        notif_priority_index = priority_levels.index(notification.get('priority', 'medium'))
        
        if notif_priority_index < min_priority_index:
            return False
        
        return True
    
    def _default_preferences(self) -> Dict:
        """Get default preferences."""
        return {
            'enabled_channels': ['email', 'in_app'],
            'quiet_hours': {'start': '22:00', 'end': '07:00'},
            'alert_types': {
                'inventory': True,
                'sales': True,
                'financial': True,
                'system': True
            },
            'min_priority': 'medium'
        }
    
    def _is_quiet_hours(self, quiet_hours: Dict) -> bool:
        """Check if current time is within quiet hours."""
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        
        start = quiet_hours['start']
        end = quiet_hours['end']
        
        # Simple time range check (doesn't handle crossing midnight perfectly)
        if start < end:
            return start <= current_time <= end
        else:
            return current_time >= start or current_time <= end


class NotificationTemplate:
    """
    Manages notification message templates.
    """
    
    def __init__(self):
        self.templates = self._init_templates()
        
    def render(self, template_name: str, data: Dict) -> Dict:
        """Render notification from template."""
        if template_name not in self.templates:
            return {'error': 'Template not found'}
        
        template = self.templates[template_name]
        
        # Replace placeholders
        subject = template['subject']
        body = template['body']
        sms = template.get('sms', body[:160])
        
        for key, value in data.items():
            placeholder = f"{{{key}}}"
            subject = subject.replace(placeholder, str(value))
            body = body.replace(placeholder, str(value))
            sms = sms.replace(placeholder, str(value))
        
        return {
            'subject': subject,
            'body': body,
            'sms': sms,
            'title': template.get('title', subject)
        }
    
    def _init_templates(self) -> Dict:
        """Initialize notification templates."""
        return {
            'low_stock': {
                'subject': 'Low Stock Alert: {product_name}',
                'title': 'Low Stock Alert',
                'body': 'Product {product_name} is running low. Current stock: {current_stock}. Reorder point: {reorder_point}.',
                'sms': 'Low stock: {product_name} ({current_stock} remaining)'
            },
            'out_of_stock': {
                'subject': 'URGENT: Out of Stock - {product_name}',
                'title': 'Out of Stock',
                'body': 'Product {product_name} is now out of stock. Immediate restocking recommended.',
                'sms': 'OUT OF STOCK: {product_name}'
            },
            'large_sale': {
                'subject': 'Large Sale Completed: {amount} TZS',
                'title': 'Large Sale',
                'body': 'A large sale of {amount} TZS was completed by {employee}. Customer: {customer}',
                'sms': 'Large sale: {amount} TZS'
            },
            'debt_exceeded': {
                'subject': 'Customer Debt Limit Exceeded: {customer}',
                'title': 'Debt Alert',
                'body': 'Customer {customer} has exceeded their credit limit. Current debt: {debt_amount} TZS. Limit: {credit_limit} TZS.',
                'sms': 'Debt alert: {customer} - {debt_amount} TZS'
            },
            'revenue_drop': {
                'subject': 'Revenue Drop Alert: {drop_percent}%',
                'title': 'Revenue Alert',
                'body': 'Daily revenue dropped by {drop_percent}% compared to average. Current: {current_revenue} TZS. Average: {avg_revenue} TZS.',
                'sms': 'Revenue down {drop_percent}%'
            }
        }
