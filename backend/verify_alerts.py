import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from intelligence.alerts import AlertManager
import logging

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

print("Running AlertManager Test...")
try:
    # Run synchronously
    stock_alert = AlertManager.check_stock_levels()
    print(f"Stock Check performed. Triggered? {stock_alert}")
    
    revenue_alert = AlertManager.check_revenue_drop()
    print(f"Revenue Check performed. Triggered? {revenue_alert}")
    
except Exception as e:
    print(f"Error: {e}")
