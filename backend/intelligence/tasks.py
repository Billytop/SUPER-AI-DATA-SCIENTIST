from celery import shared_task
from .alerts import AlertManager
from messaging.services import EmailService
from bi.engine import KPIEngine
import logging

logger = logging.getLogger(__name__)

@shared_task
def hourly_watchdog():
    """
    Runs every hour to check critical thresholds.
    """
    logger.info("Running Watchdog...")
    AlertManager.check_stock_levels()
    # AlertManager.check_revenue_drop()
    return "Watchdog cycle complete."

@shared_task
def weekly_report():
    """
    Generates and sends the Monday Morning Report.
    """
    logger.info("Generating Weekly Report...")
    data = KPIEngine.get_sales_overview(days=7)
    
    html_content = f"""
    <h1>Weekly Executive Summary</h1>
    <p>Total Revenue: <b>${data['total_revenue']}</b></p>
    <p>Have a productive week!</p>
    """
    
    # In real prod, generate PDF attachment here
    EmailService.send_report("admin@sephlighty.ai", "Monday Morning Report", html_content)
    return "Weekly Report Sent."
