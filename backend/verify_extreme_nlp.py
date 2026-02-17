import sys
import io
import os

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup Path
backend_dir = os.getcwd()
sys.path.append(backend_dir)

# Import Engine
try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
    engine = OmnibrainSaaSEngine()
    
    print("--- 🚀 EXTREME NLP VERIFICATION ---")
    
    # Test 1: Data Retrieval (Should be naturalized)
    q1 = "How many users do we have?"
    print(f"\nUser: {q1}")
    res1 = engine.process_query_v2(q1)
    print(f"AI: {res1}")
    
    # Test 2: General Chat (Should be sentient-like)
    q2 = "Who are you exactly?"
    print(f"\nUser: {q2}")
    res2 = engine.process_query_v2(q2)
    print(f"AI: {res2}")
    
    # Test 3: Forecasting (Should be conversational)
    q3 = "Predict sales for tomorrow"
    print(f"\nUser: {q3}")
    res3 = engine.process_query_v2(q3)
    print(f"AI: {res3}")

    print("\n--- ✅ VERIFICATION COMPLETE ---")

except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
