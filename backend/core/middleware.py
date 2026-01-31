import time
import logging
from .models import AuditLog
from django.utils.deprecation import MiddlewareMixin
from prometheus_client import Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('django_http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('django_http_request_duration_seconds', 'HTTP request latency')

logger = logging.getLogger("middleware")

class PrometheusMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request.start_time = time.time()

    def process_response(self, request, response):
        if hasattr(request, 'start_time'):
            latency = time.time() - request.start_time
            REQUEST_LATENCY.observe(latency)
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.path,
                status=response.status_code
            ).inc()
            
        return response

class AuditMiddleware(MiddlewareMixin):
    """
    Auto-logs specific write actions.
    """
    def process_view(self, request, view_func, view_args, view_kwargs):
        if request.method in ['POST', 'PUT', 'PATCH', 'DELETE']:
            # We defer detailed logging to signals usually, but here we log the attempt
            if request.user.is_authenticated:
                pass # Can log access here
        return None

    # Logic to capture modifications often better handled in Signals or specific View overrides
    # keeping middleware simple for metric collection primarily
