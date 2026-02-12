
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def test_phase_37():
    print("\n--- [STARTING PHASE 37 VERIFICATION: SOVEREIGN ANALYTICS] ---")
    
    engine = OmnibrainSaaSEngine()
    
    # 1. Test PDF Generation
    print("\n--- 1. Testing PDF Ledger Generation ---")
    query_pdf = "Naomba PDF ledger ya Paschal White"
    response_pdf = engine.process_query_v2(query_pdf)
    print(f"Query: '{query_pdf}'")
    print(f"Response: {response_pdf}")
    
    if "RPFL Generated" in response_pdf:
        print("[SUCCESS] PDF Generation Logic Triggered Successfully.")
    else:
        print("[FAILURE] PDF Generation Logic FAILED.")

    # 2. Test Customer Intelligence
    print("\n--- 2. Testing Customer Intelligence ---")
    query_intel = "Analyze customer Paschal White"
    response_intel = engine.process_query_v2(query_intel)
    print(f"Query: '{query_intel}'")
    # print(f"Response: {response_intel}")
    
    if "Spending Tier" in response_intel:
        print("[SUCCESS] Customer Profile Assessment Successful.")
    else:
        print("[FAILURE] Customer Profile Assessment FAILED.")

    # 3. Test Visualization
    print("\n--- 3. Testing Graph Generation ---")
    query_graph = "Show me spending graph of Paschal White"
    response_graph = engine.process_query_v2(query_graph)
    print(f"Query: '{query_graph}'")
    # print(f"Response: {response_graph}")
    
    if "CHART_DATA" in response_graph:
        print("[SUCCESS] Visualization Data Generated Successfully.")
    else:
        print("[FAILURE] Visualization Data FAILED.")

    print("\n--- [PHASE 37 VERIFICATION COMPLETE] ---")

if __name__ == "__main__":
    test_phase_37()
