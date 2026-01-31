"""
Comprehensive Audit Logger and Access Control
Full security module with auditing, permissions, encryption, and compliance.
"""

from typing import Dict, List, Optional
from datetime import datetime
import hashlib
import json


class AuditLogger:
    """
    Comprehensive audit logging for all system actions.
    """
    
    def __init__(self, storage_path: str = ''):
        self.storage_path = storage_path
        self.audit_log = []
        self.log_counter = 0
        
    def log_action(self, user_id: str, action: str, resource_type: str, resource_id: str, details: Dict = None, ip_address: str = '') -> str:
        """
        Log a user action.
        
        Args:
            user_id: User performing action
            action: Action type (create, read, update, delete, login, etc.)
            resource_type: Type of resource (customer, product, transaction, etc.)
            resource_id: ID of affected resource
            details: Additional action details
            ip_address: User's IP address
            
        Returns:
            Audit log entry ID
        """
        self.log_counter += 1
        entry_id = f"audit_{self.log_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        entry = {
            'id': entry_id,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource_type': resource_type,
            'resource_id': resource_id,
            'details': details or {},
            'ip_address': ip_address,
            'success': True
        }
        
        self.audit_log.append(entry)
        
        # In production, write to secure log file or database
        self._persist_log(entry)
        
        return entry_id
    
    def log_security_event(self, event_type: str, severity: str, description: str, details: Dict = None):
        """Log security events (failed logins, suspicious activity, etc.)."""
        entry = {
            'id': f"security_{self.log_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'type': 'security_event',
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'details': details or {}
        }
        
        self.audit_log.append(entry)
        self._persist_log(entry)
    
    def log_data_access(self, user_id: str, query_type: str, table: str, filters: Dict = None, result_count: int = 0):
        """Log database access for compliance."""
        entry = {
            'id': f"data_access_{self.log_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'type': 'data_access',
            'user_id': user_id,
            'query_type': query_type,
            'table': table,
            'filters': filters or {},
            'result_count': result_count
        }
        
        self.audit_log.append(entry)
        self._persist_log(entry)
    
    def search_logs(self, criteria: Dict, limit: int = 100) -> List[Dict]:
        """Search audit logs."""
        filtered = self.audit_log
        
        if 'user_id' in criteria:
            filtered = [log for log in filtered if log.get('user_id') == criteria['user_id']]
        
        if 'action' in criteria:
            filtered = [log for log in filtered if log.get('action') == criteria['action']]
        
        if 'resource_type' in criteria:
            filtered = [log for log in filtered if log.get('resource_type') == criteria['resource_type']]
        
        if 'start_date' in criteria:
            filtered = [log for log in filtered if log.get('timestamp', '') >= criteria['start_date']]
        
        if 'end_date' in criteria:
            filtered = [log for log in filtered if log.get('timestamp', '') <= criteria['end_date']]
        
        return filtered[-limit:]
    
    def get_user_activity(self, user_id: str, days: int = 30) -> Dict:
        """Get user activity summary."""
        user_logs = [log for log in self.audit_log if log.get('user_id') == user_id]
        
        if not user_logs:
            return {'user_id': user_id, 'activity': 'none'}
        
        actions = {}
        for log in user_logs:
            action = log.get('action', 'unknown')
            actions[action] = actions.get(action, 0) + 1
        
        return {
            'user_id': user_id,
            'total_actions': len(user_logs),
            'actions_breakdown': actions,
            'first_activity': user_logs[0]['timestamp'],
            'last_activity': user_logs[-1]['timestamp']
        }
    
    def generate_compliance_report(self, start_date: str, end_date: str) -> Dict:
        """Generate compliance audit report."""
        filtered = self.search_logs({'start_date': start_date, 'end_date': end_date}, limit=10000)
        
        users = set(log.get('user_id') for log in filtered if log.get('user_id'))
        actions = {}
        resource_types = {}
        
        for log in filtered:
            action = log.get('action', 'unknown')
            actions[action] = actions.get(action, 0) + 1
            
            resource = log.get('resource_type', 'unknown')
            resource_types[resource] = resource_types.get(resource, 0) + 1
        
        return {
            'report_period': {'start': start_date, 'end': end_date},
            'total_events': len(filtered),
            'unique_users': len(users),
            'actions_breakdown': actions,
            'resource_types': resource_types,
            'generated_at': datetime.now().isoformat()
        }
    
    def _persist_log(self, entry: Dict):
        """Persist log entry to storage (simulated)."""
        # In production, write to secure append-only log file
        pass


class PermissionManager:
    """
    Role-based access control (RBAC) system.
    """
    
    def __init__(self):
        self.roles = self._init_default_roles()
        self.user_roles = {}
        self.permissions = {}
        
    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user."""
        if role not in self.roles:
            return False
        
        self.user_roles[user_id] = role
        return True
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has permission."""
        role = self.user_roles.get(user_id)
        
        if not role:
            return False
        
        role_permissions = self.roles.get(role, {}).get('permissions', [])
        
        return permission in role_permissions or '*' in role_permissions
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for a user."""
        role = self.user_roles.get(user_id)
        
        if not role:
            return []
        
        return self.roles.get(role, {}).get('permissions', [])
    
    def create_role(self, role_name: str, permissions: List[str], description: str = ''):
        """Create custom role."""
        self.roles[role_name] = {
            'permissions': permissions,
            'description': description,
            'created_at': datetime.now().isoformat()
        }
    
    def _init_default_roles(self) -> Dict:
        """Initialize default roles."""
        return {
            'admin': {
                'permissions': ['*'],  # All permissions
                'description': 'Full system access'
            },
            'manager': {
                'permissions': [
                    'view_sales', 'view_inventory', 'view_customers',
                    'edit_products', 'edit_prices', 'view_reports',
                    'export_data'
                ],
                'description': 'Management level access'
            },
            'sales': {
                'permissions': [
                    'create_sale', 'view_sales', 'view_customers',
                    'view_products', 'view_inventory'
                ],
                'description': 'Sales staff access'
            },
            'cashier': {
                'permissions': [
                    'create_sale', 'view_products', 'view_customers'
                ],
                'description': 'Basic cashier access'
            },
            'viewer': {
                'permissions': [
                    'view_sales', 'view_products', 'view_reports'
                ],
                'description': 'Read-only access'
            }
        }


class DataEncryption:
    """
    Data encryption utilities for sensitive information.
    """
    
    def __init__(self, encryption_key: str = ''):
        self.encryption_key = encryption_key or self._generate_key()
        
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        # Simple hash-based encryption (in production use proper encryption library)
        combined = f"{data}{self.encryption_key}"
        encrypted = hashlib.sha256(combined.encode()).hexdigest()
        return encrypted
    
    def encrypt_dict(self, data: Dict, fields: List[str]) -> Dict:
        """Encrypt specific fields in a dictionary."""
        encrypted_data = data.copy()
        
        for field in fields:
            if field in encrypted_data:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))
        
        return encrypted_data
    
    def hash_password(self, password: str) -> str:
        """Hash password securely."""
        salt = self.encryption_key[:16]
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return self.hash_password(password) == hashed
    
    def _generate_key(self) -> str:
        """Generate encryption key."""
        return hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()


class ComplianceChecker:
    """
    Check compliance with regulations (GDPR, data retention, etc.).
    """
    
    def __init__(self):
        self.compliance_rules = self._init_rules()
        self.violations = []
        
    def check_data_retention(self, data_type: str, record_date: str) -> Dict:
        """Check if data exceeds retention period."""
        if data_type not in self.compliance_rules:
            return {'compliant': True, 'reason': 'No retention rule defined'}
        
        retention_days = self.compliance_rules[data_type]['retention_days']
        
        # Parse date and check age
        try:
            record_datetime = datetime.fromisoformat(record_date)
            age_days = (datetime.now() - record_datetime).days
            
            if age_days > retention_days:
                return {
                    'compliant': False,
                    'reason': f'Data exceeds retention period ({age_days} days > {retention_days} days)',
                    'action': 'archive_or_delete'
                }
            else:
                return {
                    'compliant': True,
                    'days_remaining': retention_days - age_days
                }
        except:
            return {'compliant': True, 'reason': 'Unable to parse date'}
    
    def check_gdpr_compliance(self, user_data: Dict) -> Dict:
        """Check GDPR compliance for user data."""
        issues = []
        
        # Check for consent
        if not user_data.get('consent_given'):
            issues.append('Missing user consent for data processing')
        
        # Check for encrypted sensitive fields
        sensitive_fields = ['email', 'phone', 'address']
        for field in sensitive_fields:
            if field in user_data and not user_data.get(f'{field}_encrypted'):
                issues.append(f'Sensitive field {field} not encrypted')
        
        # Check for data access request capability
        if not user_data.get('can_export_data'):
            issues.append('User cannot export their data (GDPR right)')
        
        return {
            'compliant': len(issues) == 0,
            'issues': issues
        }
    
    def generate_compliance_score(self, system_data: Dict) -> Dict:
        """Generate overall compliance score."""
        total_checks = 5
        passed_checks = 0
        
        checks = {
            'audit_logging_enabled': system_data.get('audit_enabled', False),
            'encryption_enabled': system_data.get('encryption_enabled', False),
            'access_control_configured': system_data.get('rbac_enabled', False),
            'data_retention_policy': system_data.get('retention_policy', False),
            'user_consent_tracking': system_data.get('consent_tracking', False)
        }
        
        passed_checks = sum(1 for v in checks.values() if v)
        score = (passed_checks / total_checks) * 100
        
        return {
            'score': score,
            'checks': checks,
            'level': 'excellent' if score >= 90 else 'good' if score >= 70 else 'needs_improvement'
        }
    
    def _init_rules(self) -> Dict:
        """Initialize compliance rules."""
        return {
            'customer_data': {'retention_days': 2555},  # 7 years
            'transaction_data': {'retention_days': 2555},
            'audit_logs': {'retention_days': 365},
            'session_logs': {'retention_days': 90}
        }
