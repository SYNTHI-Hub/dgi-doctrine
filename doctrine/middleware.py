import time
import logging
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)


class APILoggingMiddleware(MiddlewareMixin):
    """
    Middleware pour logger les requêtes API
    """

    def process_request(self, request):
        if request.path.startswith('/api/'):
            request.start_time = time.time()
            logger.info(f"API Request: {request.method} {request.path} - User: {request.user}")

    def process_response(self, request, response):
        if hasattr(request, 'start_time') and request.path.startswith('/api/'):
            duration = time.time() - request.start_time
            logger.info(
                f"API Response: {request.method} {request.path} - "
                f"Status: {response.status_code} - Duration: {duration:.3f}s"
            )
        return response


class RateLimitMiddleware(MiddlewareMixin):
    """
    Middleware simple de limitation de taux pour l'API
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.request_counts = {}

    def __call__(self, request):
        if request.path.startswith('/api/') and request.user.is_authenticated:
            user_id = str(request.user.id)
            current_time = int(time.time())

            self.request_counts = {
                uid: [(timestamp, count) for timestamp, count in requests
                      if current_time - timestamp < 60]
                for uid, requests in self.request_counts.items()
            }

            user_requests = self.request_counts.get(user_id, [])
            total_requests = sum(count for _, count in user_requests)

            rate_limit = getattr(request.user, 'api_rate_limit', 1000)

            if total_requests >= rate_limit:
                from django.http import JsonResponse
                return JsonResponse(
                    {'error': 'Rate limit exceeded'},
                    status=429
                )

            if user_id not in self.request_counts:
                self.request_counts[user_id] = []
            self.request_counts[user_id].append((current_time, 1))

        return self.get_response(request)