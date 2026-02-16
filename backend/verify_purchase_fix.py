import os
import sys
import json
import logging

# Setup Path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import Engine
from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def test_purchase_fixes():
    print("\n--- INITIALIZING OMNIBRAIN ENGINE ---")
    engine = OmnibrainSaaSEngine()
    
    # Test 1: Global Chart (No Customer specified)
    print("\n\n[TEST 1] Testing Global Chart Generation for 2026...")
    query_chart = "draw graph of purchases 2026"
    response_chart = engine.process_query_v2(query_chart)
    print(f"QUERY: {query_chart}")
    print(f"RESPONSE LENGTH: {len(str(response_chart))}")
    if "[CHART_DATA]" in str(response_chart):
        print("✅ SUCCESS: Global Chart Data detected.")
        # Extract and parse JSON to verify structure
        try:
            json_str = str(response_chart).split("[CHART_DATA]:")[1].strip()
            data = json.loads(json_str)
            print(f"   Chart Title: {data.get('title')}")
            print(f"   Datasets: {len(data.get('datasets', []))}")
        except Exception as e:
            print(f"⚠️  Chart JSON Parse Error: {e}")
    else:
        print("❌ FAILURE: No [CHART_DATA] found.")
        print(response_chart)

    # Test 2: Deep Purchase Intelligence (Ranges)
    print("\n\n[TEST 2] Testing Deep Purchase Intelligence (Ranges)...")
    query_intel = "analyze purchases according to range and more bigger"
    response_intel = engine.process_query_v2(query_intel)
    print(f"QUERY: {query_intel}")
    
    if "PURCHASE RANGES" in str(response_intel) or "MADARAJA YA MANUNUZI" in str(response_intel):
        print("✅ SUCCESS: Range-based analysis detected.")
        print("--- SNIPPET ---")
        print(str(response_intel)[:500] + "...")
    else:
        print("❌ FAILURE: Range-based section not found.")
        print(f"   Full Response: {str(response_intel)[:500]}")

    # Test 3: Reproduce 'sku' Error
    print("\n\n[TEST 3] Reproducing 'sku' Error...")
    query_error = "most purchased products frompaschal white dodoma my custumer"
    response_error = engine.process_query_v2(query_error)
    print(f"QUERY: {query_error}")
    print(f"RESPONSE: {response_error}")

    # Test 4: List All
    print("\n\n[TEST 4] Testing 'list order zake zote'...")
    query_list = "list order zake zote"
    response_list = engine.process_query_v2(query_list)
    print(f"QUERY: {query_list}")
    print(f"RESPONSE: {response_list}")

    # Test 5: Reproduce 'most purchases products on2026'
    print("\n\n[TEST 5] Reproducing 'most purchases products on2026'...")
    query_error_2 = "most purchases products on2026"
    response_error_2 = engine.process_query_v2(query_error_2)
    print(f"QUERY: {query_error_2}")
    print(f"RESPONSE: {response_error_2}")

if __name__ == "__main__":
    test_purchase_fixes()
