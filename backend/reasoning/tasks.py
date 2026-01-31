from celery import shared_task
from django.core.management import call_command
from bi.engine import KPIEngine
import logging

logger = logging.getLogger(__name__)

@shared_task
def reindex_knowledge_base():
    """
    Re-runs embedding for Schema and Knowledge Facts.
    Should run nightly.
    """
    logger.info("Starting Knowledge Base Re-indexing...")
    try:
        call_command('embed_schema')
        return "Schema & Knowledge Re-indexed."
    except Exception as e:
        return f"Re-indexing failed: {e}"

@shared_task
def precompute_analytics():
    """
    Runs heavy BI queries and caches them.
    """
    logger.info("Pre-computing Analytics...")
    try:
        # We just call the engine, assuming it uses caching internally 
        # (or we could explicitly set cache here)
        data = KPIEngine.get_sales_overview(days=30)
        return f"Precompted Sales: ${data['total_revenue']}"
    except Exception as e:
        return f"Pre-compute failed: {e}"
