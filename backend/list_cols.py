import os
import sys

# Setup Path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import Engine
from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def list_columns():
    engine = OmnibrainSaaSEngine()
    res = engine._execute_erp_query("DESCRIBE transactions")
    cols = [r['Field'] for r in res]
    print(", ".join(cols))

if __name__ == "__main__":
    list_columns()
