import sys
import os
sys.path.append(os.getcwd())
try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
    engine = OmnibrainSaaSEngine()
    q = "compare employee njukuibilali and shakiraismail"
    entities = engine._resolve_entities(q, "users")
    print(f"Query: {q}")
    print(f"Resolved Entities: {[e['username'] for e in entities]}")
    
    if len(entities) >= 2:
        print("SUCCESS: Both users resolved despite typos.")
        print(engine._compare_users(entities))
    else:
        print("FAILURE: Could not resolve both users.")
except Exception as e:
    print(f"Error: {e}")
