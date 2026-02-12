
import os
import sys

# Ensure the backend path is included
sys.path.append(os.path.join(os.getcwd(), "backend/laravel_modules/ai_brain"))

# Absolute imports
from omnibrain_saas_engine import OmnibrainSaaSEngine

# Fix for Windows Unicode printing
sys.stdout.reconfigure(encoding='utf-8')

def debug_paschal():
    print("\n--- [STARTING DEBUG: PASCHAL WHITE QUERY] ---")
    engine = OmnibrainSaaSEngine()
    
    queries = [
        "deni la paschal white dododma ni ngapi",
        "naomba ledger ya pashal white dodoma",
        "kwann deni ni kubwa",
        "naomba ledger paschl whte ddoma"
    ]
    
    for q in queries:
        print(f"\nQUERY: '{q}'")
        
        # 1. Test Entity Resolution directly
        print("1. Entity Resolution Check:")
        entities = engine._resolve_entities(q.lower())
        print(f"   Found: {entities}")
        
        # 2. Test Intent Routing
        print("2. Intent Analysis:")
        # We need to see what _route_intent thinks (simulated via strict regex check first)
        if hasattr(engine, '_clean_query'):
            clean_q = engine._clean_query(q)
            print(f"   Clean Query: {clean_q}")
        
        # 3. Process Query V2
        print("3. Full Processing:")
        try:
            response = engine.process_query_v2(q)
            # Truncate long responses for readability
            print(f"   Response: {response[:200]}...")
        except Exception as e:
            print(f"   ERROR: {e}")

if __name__ == "__main__":
    debug_paschal()
