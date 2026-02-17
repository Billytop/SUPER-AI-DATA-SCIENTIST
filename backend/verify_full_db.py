import sys
import io
import os
import time

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup Path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Import Engine
try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
except ImportError as e:
    print(f"❌ Could not import Engine: {e}")
    sys.exit(1)

def verify_full_db():
    print("--- VERIFYING FULL DB AWARENESS (PHASE 5) ---")
    sys.stdout.flush()
    
    # Initialize
    try:
        engine = OmnibrainSaaSEngine()
        print("✅ Engine Initialized.")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Engine Init Failed: {e}")
        sys.stdout.flush()
        return

    # 1. Check Schema Map
    print("\n1. Checking Schema Map...")
    sys.stdout.flush()
    try:
        tables = list(engine.schema_map.keys())
        print(f"📊 Found {len(tables)} tables.")
        print(f"📝 Sample tables: {tables[:5]}")
        sys.stdout.flush()
        
        if len(tables) > 5:
            print("✅ SUCCESS: Schema mapped successfully.")
        else:
            print("⚠️ WARNING: Very few tables found. Check DB connection.")
        sys.stdout.flush()
            
    except Exception as e:
        print(f"❌ Schema Check Failed: {e}")
        sys.stdout.flush()

    # 2. Test Dynamic Retrieval (Direct)
    print("\n2. Testing Dynamic Data Retrieval (Target: 'products')...")
    sys.stdout.flush()
    try:
        # We assume 'products' table exists.
        response = engine._dynamic_data_retrieval("show me products")
        print(f"📁 Dynamic Output:\n---\n{response}\n---")
        sys.stdout.flush()
        
        if response and "Universal Access" in response:
            print("✅ SUCCESS: Dynamic retrieval worked.")
        else:
            print("❌ FAILURE: Dynamic retrieval returned None or wrong format.")
        sys.stdout.flush()

    except Exception as e:
         print(f"❌ Dynamic Retrieval Failed: {e}")
         sys.stdout.flush()

if __name__ == "__main__":
    verify_full_db()
