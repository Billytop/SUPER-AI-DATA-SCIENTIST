import os
import sys
import json

# Setup Path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import Engine
from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def describe_table():
    print("--- DESCRIBING TRANSACTIONS TABLE ---")
    engine = OmnibrainSaaSEngine()
    res = engine._execute_erp_query("DESCRIBE transactions")
    for r in res:
        print(f"{r['Field']}: {r['Type']}")

if __name__ == "__main__":
    describe_table()
