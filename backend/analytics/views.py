from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .engine import KPIEngine
from .ml import SalesForecaster

class AnalyticsDashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Descriptive (What happened)
        sales_data = KPIEngine.get_sales_overview(days=30)
        top_products = KPIEngine.get_top_products()
        
        # Predictive (What will happen)
        forecast = SalesForecaster.predict_next_30_days()
        
        return Response({
            "kpi": {
                "total_revenue_30d": sales_data['total_revenue'],
                "avg_daily_sales": sales_data['total_revenue'] / 30
            },
            "charts": {
                "daily_sales": sales_data['daily_sales'],
                "top_products": top_products,
                "forecast_30d": forecast
            }
        })
