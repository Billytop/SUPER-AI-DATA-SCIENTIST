from bi.engine import KPIEngine
from messaging.services import WhatsAppService, EmailService
import logging

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Watchdog service that checks business health and triggers alerts.
    """
    
    @staticmethod
    def check_stock_levels():
        """
        Checks for low stock and alerts via WhatsApp.
        """
        # In a real scenario, KPIEngine would return specific low-stock items
        # For prototype, we mock the check
        low_stock_items = [] # KPIEngine.get_low_stock()
        
        if low_stock_items:
            msg = "⚠️ Low Stock Alert:\n" + "\n".join([f"- {item.name}: {item.qty}" for item in low_stock_items])
            # Assuming we have a configured admin number
            admin_number = "1234567890" 
            WhatsAppService.send_message(admin_number, msg)
            return True
        return False

    @staticmethod
    def check_revenue_drop():
        """
        Alerts if revenue drops significantly compared to last week.
        """
        current_sales = KPIEngine.get_sales_overview(days=7)['total_revenue']
        # mock previous
        previous_sales = current_sales * 1.1 # Simulate 10% drop
        
        if current_sales < previous_sales * 0.8: # 20% drop
            EmailService.send_report(
                "admin@sephlighty.ai",
                "⚠️ Revenue Alert",
                f"<h1>Revenue Drop Detected</h1><p>This week: ${current_sales}</p><p>Last week: ${previous_sales}</p>"
            )
            return True
        return False
