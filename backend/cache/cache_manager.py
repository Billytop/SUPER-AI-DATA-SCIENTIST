"""
Multi-Level Caching System
Redis-style caching with TTL, LRU eviction, and performance monitoring.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib
import json


class Cache Level:
    """Single cache level."""
    
    def __init__(self, name: str, max_size: int = 1000, ttl: int = 3600):
        self.name = name
        self.max_size = max_size
        self.default_ttl = ttl
        self.store = OrderedDict()
        self.metadata = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.store:
            self.stats['misses'] += 1
            return None
        
        # Check expiration
        if self._is_expired(key):
            self.delete(key)
            self.stats['misses'] += 1
            return None
        
        # Move to end (LRU)
        self.store.move_to_end(key)
        self.stats['hits'] += 1
        
        return self.store[key]
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache."""
        # Evict if at capacity
        if len(self.store) >= self.max_size and key not in self.store:
            self._evict_oldest()
        
        self.store[key] = value
        self.store.move_to_end(key)
        
        self.metadata[key] = {
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=ttl or self.default_ttl),
            'access_count': 0,
            'size_bytes': len(str(value))
        }
        
        self.stats['sets'] += 1
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.store:
            del self.store[key]
            del self.metadata[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache entries."""
        self.store.clear()
        self.metadata.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'name': self.name,
            'size': len(self.store),
            'max_size': self.max_size,
            'hit_rate': hit_rate,
            **self.stats
        }
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        if key not in self.metadata:
            return True
        
        return datetime.now() > self.metadata[key]['expires_at']
    
    def _evict_oldest(self):
        """Evict oldest entry (LRU)."""
        if self.store:
            oldest_key = next(iter(self.store))
            self.delete(oldest_key)
            self.stats['evictions'] += 1


class MultiLevelCache:
    """
    Multi-level caching system with L1 (memory), L2 (local), L3 (distributed).
    """
    
    def __init__(self):
        self.levels = {
            'L1': CacheLevel('L1_Memory', max_size=500, ttl=300),     # 5 min, small fast cache
            'L2': CacheLevel('L2_Local', max_size=5000, ttl=3600),    # 1 hour, larger cache
            'L3': CacheLevel('L3_Shared', max_size=50000, ttl=86400)  # 24 hours, huge cache
        }
        self.cache_policies = {}
        
    def get(self, key: str, policy: str = 'L1') -> Optional[Any]:
        """Get value from cache, checking all levels."""
        # Try L1 first (fastest)
        value = self.levels['L1'].get(key)
        if value is not None:
            return value
        
        # Try L2
        value = self.levels['L2'].get(key)
        if value is not None:
            # Promote to L1
            self.levels['L1'].set(key, value)
            return value
        
        # Try L3
        value = self.levels['L3'].get(key)
        if value is not None:
            # Promote to L2 and L1
            self.levels['L2'].set(key, value)
            self.levels['L1'].set(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, levels: List[str] = None, ttl: int = None):
        """Set value in specified cache levels."""
        if levels is None:
            levels = ['L1', 'L2', 'L3']
        
        for level in levels:
            if level in self.levels:
                self.levels[level].set(key, value, ttl)
    
    def delete(self, key: str):
        """Delete from all cache levels."""
        for level in self.levels.values():
            level.delete(key)
    
    def clear_all(self):
        """Clear all cache levels."""
        for level in self.levels.values():
            level.clear()
    
    def get_overall_stats(self) -> Dict:
        """Get statistics for all cache levels."""
        return {
            level_name: level.get_stats()
            for level_name, level in self.levels.items()
        }


class QueryResultCache:
    """
    Specialized cache for database query results.
    """
    
    def __init__(self):
        self.cache = MultiLevelCache()
        self.query_stats = {}
        
    def cache_query_result(self, query: str, params: Dict, result: Any, ttl: int = 300):
        """Cache query result."""
        cache_key = self._generate_query_key(query, params)
        self.cache.set(cache_key, result, levels=['L1', 'L2'], ttl=ttl)
        
        # Track query stats
        if cache_key not in self.query_stats:
            self.query_stats[cache_key] = {
                'query': query,
                'cache_hits': 0,
                'cache_misses': 0,
                'total_executions': 0
            }
        
        self.query_stats[cache_key]['total_executions'] += 1
    
    def get_cached_result(self, query: str, params: Dict) -> Optional[Any]:
        """Get cached query result."""
        cache_key = self._generate_query_key(query, params)
        result = self.cache.get(cache_key)
        
        if cache_key in self.query_stats:
            if result is not None:
                self.query_stats[cache_key]['cache_hits'] += 1
            else:
                self.query_stats[cache_key]['cache_misses'] += 1
        
        return result
    
    def invalidate_by_table(self, table_name: str):
        """Invalidate all cached queries for a table."""
        # In production, would track table dependencies
        # For now, clear all
        self.cache.clear_all()
    
    def get_query_stats(self) -> List[Dict]:
        """Get query cache statistics."""
        stats = []
        for key, data in self.query_stats.items():
            hit_rate = (data['cache_hits'] / (data['cache_hits'] + data['cache_misses']) * 100) \
                if (data['cache_hits'] + data['cache_misses']) > 0 else 0
            
            stats.append({
                'query': data['query'][:100],  # Truncate long queries
                'hit_rate': hit_rate,
                **data
            })
        
        return sorted(stats, key=lambda x: x['cache_hits'], reverse=True)
    
    def _generate_query_key(self, query: str, params: Dict) -> str:
        """Generate cache key from query and parameters."""
        param_str = json.dumps(params, sort_keys=True) if params else ''
        combined = f"{query}:{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()


class PerformanceMonitor:
    """
    Monitors system performance metrics.
    """
    
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'query_times': [],
            'cache_operations': [],
            'errors': []
        }
        self.alerts = []
        self.thresholds = {
            'response_time_ms': 1000,
            'query_time_ms': 500,
            'error_rate_percent': 5.0
        }
        
    def record_response_time(self, endpoint: str, duration_ms: float):
        """Record API response time."""
        self.metrics['response_times'].append({
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'duration_ms': duration_ms
        })
        
        # Check threshold
        if duration_ms > self.thresholds['response_time_ms']:
            self._create_alert('slow_response', f'Slow response: {endpoint} took {duration_ms}ms')
    
    def record_query_time(self, query: str, duration_ms: float):
        """Record database query time."""
        self.metrics['query_times'].append({
            'timestamp': datetime.now().isoformat(),
            'query': query[:100],
            'duration_ms': duration_ms
        })
        
        if duration_ms > self.thresholds['query_time_ms']:
            self._create_alert('slow_query', f'Slow query took {duration_ms}ms')
    
    def record_error(self, error_type: str, message: str):
        """Record error."""
        self.metrics['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': message
        })
    
    def get_performance_summary(self, minutes: int = 60) -> Dict:
        """Get performance summary for last N minutes."""
        cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        
        # Filter recent metrics
        recent_responses = [m for m in self.metrics['response_times'] 
                          if m['timestamp'] >= cutoff]
        recent_queries = [m for m in self.metrics['query_times']
                         if m['timestamp'] >= cutoff]
        recent_errors = [m for m in self.metrics['errors']
                        if m['timestamp'] >= cutoff]
        
        # Calculate averages
        avg_response_time = sum(m['duration_ms'] for m in recent_responses) / len(recent_responses) \
            if recent_responses else 0
        
        avg_query_time = sum(m['duration_ms'] for m in recent_queries) / len(recent_queries) \
            if recent_queries else 0
        
        return {
            'period_minutes': minutes,
            'total_requests': len(recent_responses),
            'avg_response_time_ms': avg_response_time,
            'avg_query_time_ms': avg_query_time,
            'total_errors': len(recent_errors),
            'error_rate': (len(recent_errors) / len(recent_responses) * 100) \
                if recent_responses else 0,
            'active_alerts': len(self.alerts)
        }
    
    def get_slowest_endpoints(self, limit: int = 10) -> List[Dict]:
        """Get slowest endpoints."""
        endpoints = {}
        
        for metric in self.metrics['response_times']:
            endpoint = metric['endpoint']
            if endpoint not in endpoints:
                endpoints[endpoint] = []
            endpoints[endpoint].append(metric['duration_ms'])
        
        # Calculate averages
        endpoint_stats = []
        for endpoint, times in endpoints.items():
            endpoint_stats.append({
                'endpoint': endpoint,
                'avg_time_ms': sum(times) / len(times),
                'max_time_ms': max(times),
                'request_count': len(times)
            })
        
        return sorted(endpoint_stats, key=lambda x: x['avg_time_ms'], reverse=True)[:limit]
    
    def get_slowest_queries(self, limit: int = 10) -> List[Dict]:
        """Get slowest database queries."""
        queries = {}
        
        for metric in self.metrics['query_times']:
            query = metric['query']
            if query not in queries:
                queries[query] = []
            queries[query].append(metric['duration_ms'])
        
        query_stats = []
        for query, times in queries.items():
            query_stats.append({
                'query': query,
                'avg_time_ms': sum(times) / len(times),
                'max_time_ms': max(times),
                'execution_count': len(times)
            })
        
        return sorted(query_stats, key=lambda x: x['avg_time_ms'], reverse=True)[:limit]
    
    def _create_alert(self, alert_type: str, message: str):
        """Create performance alert."""
        self.alerts.append({
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message
        })
        
        # Keep only last 100 alerts
        self.alerts = self.alerts[-100:]


class CacheWarmer:
    """
    Pre-loads cache with frequently accessed data.
    """
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.warming_jobs = []
        
    def warm_cache(self, data_loader: callable, cache_keys: List[str], ttl: int = 3600):
        """Pre-load cache with data."""
        warmed_count = 0
        
        for key in cache_keys:
            try:
                data = data_loader(key)
                if data is not None:
                    self.cache.set(key, data, levels=['L2', 'L3'], ttl=ttl)
                    warmed_count += 1
            except Exception as e:
                continue
        
        return {
            'keys_warmed': warmed_count,
            'total_keys': len(cache_keys),
            'timestamp': datetime.now().isoformat()
        }
    
    def schedule_warming(self, name: str, data_loader: callable, cache_keys: List[str], interval_minutes: int = 60):
        """Schedule periodic cache warming."""
        job = {
            'name': name,
            'data_loader': data_loader,
            'cache_keys': cache_keys,
            'interval_minutes': interval_minutes,
            'last_run': None,
            'next_run': datetime.now()
        }
        
        self.warming_jobs.append(job)
        return job
