import os
import sys
import json
import logging

# Setup Path - Match verify_purchase_fix.py style
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import Engine (Assuming it handles Django setup internally or doesn't need it for imports)
try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
except ImportError:
    # If running directly in backend, we might need to add current dir
    sys.path.append(os.getcwd())
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def test_customer_refinements():
    print("--- INITIALIZING OMNIBRAIN ENGINE ---")
    engine = OmnibrainSaaSEngine()
    
    # Test 1: Advanced Customer Info
    print("\n\n[TEST 1] Advanced Customer Info (Paschal White)...")
    query_info = "give me customer info analysis for Paschal White"
    response_info = engine.process_query_v2(query_info)
    print(f"QUERY: {query_info}")
    print(f"RESPONSE: {response_info}")
    
    if "Super Advanced Intelligence" in response_info:
        print("✅ SUCCESS: Advanced Customer Report triggered.")
    else:
        print("❌ FAILURE: Standard report returned instead.")

    # Test 2: Debt (Cumulative)
    print("\n\n[TEST 2] Debt Check (Paschal White)...")
    query_debt = "deni la Paschal White"
    response_debt = engine.process_query_v2(query_debt)
    print(f"QUERY: {query_debt}")
    print(f"RESPONSE: {response_debt}")
    
    # Test 3: Date Range List
    print("\n\n[TEST 3] Date Range List (from 2026-01-01 to 2026-02-01)...")
    query_range = "list orders from 2026-01-01 to 2026-02-01"
    response_range = engine.process_query_v2(query_range)
    print(f"QUERY: {query_range}")
    print(f"RESPONSE: {response_range}")
    
    if "LIST YA" in response_range or "Jumla ya" in response_range:
        print("✅ SUCCESS: Date range query processed.")
    else:
        print("❌ FAILURE: Date range query failed.")

if __name__ == "__main__":
    test_customer_refinements()
