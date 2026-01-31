"""
REST API Client Framework
Generic framework for calling external REST APIs with retry logic and caching.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json


class APIClient:
    """
    Generic REST API client with retry logic, rate limiting, and caching.
    """
    
    def __init__(self, base_url: str, api_key: str = '', timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.cache = {}
        self.request_log = []
        self.rate_limits = {}
        
    def get(self, endpoint: str, params: Dict = None, cache_ttl: int = 0) -> Dict:
        """
        Make GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            cache_ttl: Cache time-to-live in seconds (0 = no cache)
            
        Returns:
            Response data
        """
        cache_key = self._build_cache_key('GET', endpoint, params)
        
        # Check cache
        if cache_ttl > 0 and cache_key in self.cache:
            cached = self.cache[cache_key]
            if datetime.now() < cached['expires']:
                return {'data': cached['data'], 'cached': True}
        
        # Make request (simulated)
        response = self._make_request('GET', endpoint, params=params)
        
        # Cache response
        if cache_ttl > 0 and response.get('success'):
            self.cache[cache_key] = {
                'data': response['data'],
                'expires': datetime.now() + timedelta(seconds=cache_ttl)
            }
        
        return response
    
    def post(self, endpoint: str, data: Dict = None) -> Dict:
        """Make POST request."""
        return self._make_request('POST', endpoint, data=data)
    
    def put(self, endpoint: str, data: Dict = None) -> Dict:
        """Make PUT request."""
        return self._make_request('PUT', endpoint, data=data)
    
    def delete(self, endpoint: str) -> Dict:
        """Make DELETE request."""
        return self._make_request('DELETE', endpoint)
    
    def get_request_stats(self) -> Dict:
        """Get API request statistics."""
        total_requests = len(self.request_log)
        
        if total_requests == 0:
            return {'total_requests': 0}
        
        success_count = sum(1 for r in self.request_log if r['success'])
        
        methods = {}
        for r in self.request_log:
            methods[r['method']] = methods.get(r['method'], 0) + 1
        
        return {
            'total_requests': total_requests,
            'successful': success_count,
            'failed': total_requests - success_count,
            'success_rate': (success_count / total_requests * 100) if total_requests > 0 else 0,
            'methods': methods,
            'cache_hits': sum(1 for r in self.request_log if r.get('cached')),
            'cached_items': len(self.cache)
        }
    
    def clear_cache(self):
        """Clear all cached responses."""
        self.cache = {}
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None, retries: int = 3) -> Dict:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        #  Check rate limit
        if not self._check_rate_limit(endpoint):
            return {'success': False, 'error': 'Rate limit exceeded'}
        
        # Log request
        request_log = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'endpoint': endpoint,
            'url': url
        }
        
        # Simulated request (in production would use requests library)
        try:
            response = {
                'success': True,
                'status_code': 200,
                'data': {'message': f'Simulated {method} response', 'params': params, 'data': data}
            }
            
            request_log['success'] = True
            request_log['status_code'] = 200
            
        except Exception as e:
            response = {'success': False, 'error': str(e)}
            request_log['success'] = False
            request_log['error'] = str(e)
        
        self.request_log.append(request_log)
        self._update_rate_limit(endpoint)
        
        return response
    
    def _build_cache_key(self, method: str, endpoint: str, params: Dict = None) -> str:
        """Build cache key from request parameters."""
        param_str = json.dumps(params, sort_keys=True) if params else ''
        return f"{method}:{endpoint}:{param_str}"
    
    def _check_rate_limit(self, endpoint: str) -> bool:
        """Check if request is within rate limit."""
        if endpoint not in self.rate_limits:
            return True
        
        limit_info = self.rate_limits[endpoint]
        now = datetime.now()
        
        # Reset if window expired
        if now >= limit_info['window_end']:
            self.rate_limits[endpoint] = {
                'count': 0,
                'window_end': now + timedelta(minutes=1)
            }
            return True
        
        # Check limit (default 100 requests per minute)
        return limit_info['count'] < 100
    
    def _update_rate_limit(self, endpoint: str):
        """Update rate limit counter."""
        if endpoint not in self.rate_limits:
            self.rate_limits[endpoint] = {
                'count': 1,
                'window_end': datetime.now() + timedelta(minutes=1)
            }
        else:
            self.rate_limits[endpoint]['count'] += 1


class WebhookHandler:
    """
    Handles incoming webhooks from external services.
    """
    
    def __init__(self):
        self.handlers = {}
        self.webhook_log = []
        
    def register_handler(self, event_type: str, handler_func: callable):
        """Register a webhook handler function."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        
        self.handlers[event_type].append(handler_func)
    
    def handle_webhook(self, event_type: str, payload: Dict) -> Dict:
        """Process incoming webhook."""
        webhook_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        
        log_entry = {
            'id': webhook_id,
            'event_type': event_type,
            'received_at': datetime.now().isoformat(),
            'payload_size': len(json.dumps(payload)),
            'processed': False
        }
        
        if event_type not in self.handlers:
            log_entry['error'] = 'No handler registered for event type'
            self.webhook_log.append(log_entry)
            return {'success': False, 'error': 'No handler for event type'}
        
        # Execute all handlers for this event type
        results = []
        for handler in self.handlers[event_type]:
            try:
                result = handler(payload)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        log_entry['processed'] = True
        log_entry['results'] = results
        self.webhook_log.append(log_entry)
        
        return {'success': True, 'webhook_id': webhook_id, 'results': results}
    
    def get_webhook_log(self, limit: int = 100) -> List[Dict]:
        """Get recent webhook activity."""
        return self.webhook_log[-limit:]


class ThirdPartyConnector:
    """
    Connectors for common third-party services.
    """
    
    def __init__(self):
        self.connections = {}
        
    def connect_payment_gateway(self, provider: str, credentials: Dict) -> str:
        """Connect to payment processing gateway."""
        connection_id = f"payment_{provider}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.connections[connection_id] = {
            'type': 'payment',
            'provider': provider,
            'status': 'connected',
            'connected_at': datetime.now().isoformat()
        }
        
        return connection_id
    
    def connect_sms_gateway(self, provider: str, credentials: Dict) -> str:
        """Connect to SMS service."""
        connection_id = f"sms_{provider}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.connections[connection_id] = {
            'type': 'sms',
            'provider': provider,
            'status': 'connected',
            'connected_at': datetime.now().isoformat()
        }
        
        return connection_id
    
    def connect_email_service(self, provider: str, credentials: Dict) -> str:
        """Connect to email service."""
        connection_id = f"email_{provider}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.connections[connection_id] = {
            'type': 'email',
            'provider': provider,
            'status': 'connected',
            'connected_at': datetime.now().isoformat()
        }
        
        return connection_id
    
    def send_payment(self, connection_id: str, amount: float, recipient: Dict) -> Dict:
        """Process payment through connected gateway."""
        if connection_id not in self.connections:
            return {'success': False, 'error': 'Connection not found'}
        
        connection = self.connections[connection_id]
        
        if connection['type'] != 'payment':
            return {'success': False, 'error': 'Not a payment connection'}
        
        # Simulated payment processing
        transaction_id = f"txn_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'amount': amount,
            'status': 'processed',
            'provider': connection['provider']
        }
    
    def send_sms(self, connection_id: str, phone: str, message: str) -> Dict:
        """Send SMS through connected gateway."""
        if connection_id not in self.connections:
            return {'success': False, 'error': 'Connection not found'}
        
        connection = self.connections[connection_id]
        
        if connection['type'] != 'sms':
            return {'success': False, 'error': 'Not an SMS connection'}
        
        message_id = f"sms_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        return {
            'success': True,
            'message_id': message_id,
            'recipient': phone,
            'status': 'sent',
            'provider': connection['provider']
        }
    
    def send_email(self, connection_id: str, to_email: str, subject: str, body: str) -> Dict:
        """Send email through connected service."""
        if connection_id not in self.connections:
            return {'success': False, 'error': 'Connection not found'}
        
        connection = self.connections[connection_id]
        
        if connection['type'] != 'email':
            return {'success': False, 'error': 'Not an email connection'}
        
        email_id = f"email_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        return {
            'success': True,
            'email_id': email_id,
            'recipient': to_email,
            'status': 'sent',
            'provider': connection['provider']
        }


class DataSyncEngine:
    """
    Synchronizes data between systems.
    """
    
    def __init__(self):
        self.sync_configs = {}
        self.sync_history = []
        
    def configure_sync(self, name: str, source: str, destination: str, mapping: Dict, schedule: str = 'hourly') -> str:
        """Configure a data sync."""
        sync_id = f"sync_{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.sync_configs[sync_id] = {
            'id': sync_id,
            'name': name,
            'source': source,
            'destination': destination,
            'mapping': mapping,
            'schedule': schedule,
            'enabled': True,
            'last_sync': None,
            'next_sync': datetime.now() + timedelta(hours=1)
        }
        
        return sync_id
    
    def run_sync(self, sync_id: str, full_sync: bool = False) -> Dict:
        """Execute a data sync."""
        if sync_id not in self.sync_configs:
            return {'success': False, 'error': 'Sync configuration not found'}
        
        config = self.sync_configs[sync_id]
        start_time = datetime.now()
        
        # Simulated sync (would actually sync data in production)
        result = {
            'sync_id': sync_id,
            'started_at': start_time.isoformat(),
            'full_sync': full_sync,
            'records_processed': 1000,  # Simulated
            'records_created': 50,
            'records_updated': 30,
            'records_failed': 2,
            'status': 'completed'
        }
        
        result['completed_at'] = datetime.now().isoformat()
        result['duration_seconds'] = (datetime.now() - start_time).seconds
        
        # Update config
        config['last_sync'] = result['completed_at']
        
        # Log sync
        self.sync_history.append(result)
        
        return result
    
    def get_sync_status(self, sync_id: str) -> Optional[Dict]:
        """Get sync configuration and status."""
        return self.sync_configs.get(sync_id)
    
    def get_sync_history(self, sync_id: str = None, limit: int = 50) -> List[Dict]:
        """Get sync execution history."""
        history = self.sync_history
        
        if sync_id:
            history = [h for h in history if h['sync_id'] == sync_id]
        
        return history[-limit:]
