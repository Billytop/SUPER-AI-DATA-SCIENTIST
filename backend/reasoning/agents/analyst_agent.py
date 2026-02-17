import sys
import io
import os
import logging
from datetime import datetime

# Force UTF-8 for Windows console handled by runner
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/reasoning/agents
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) # backend
sys.path.append(backend_dir)

# Import Engine
try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
except ImportError as e:
    print(f"❌ Could not import Engine: {e}")
    sys.exit(1)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_analyst():
    print("--- 🧠 STARTING ANALYST AGENT (NIGHTLY REPORT) ---")
    
    try:
        engine = OmnibrainSaaSEngine()
        logger.info("Omnibrain connected.")
        
        # Run Debrief
        report = engine._run_nightly_debrief()
        
        print("\n" + "="*40)
        print(report)
        print("="*40 + "\n")
        
        # In a real deployment, this would send an SMS/Email/Slack message
        logger.info("Debrief generated successfully.")
        
    except Exception as e:
        logger.error(f"Analyst Failed: {e}")
        print(f"❌ Analyst Crashed: {e}")

if __name__ == "__main__":
    run_analyst()
