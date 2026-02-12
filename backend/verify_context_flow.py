
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def test_context_flow():
    print("\n--- [STARTING CONTEXT FLOW VERIFICATION] ---")
    
    engine = OmnibrainSaaSEngine()
    
    # 1. Establish Context (Paschal White)
    print("\n--- 1. Establishing Context ---")
    query_1 = "Deni la Paschal White"
    response_1 = engine.process_query_v2(query_1)
    print(f"Query: '{query_1}'")
    print(f"Response: {response_1}")
    
    if "PASCHAL WHITE" in response_1.upper():
         print("[SUCCESS] Context Established.")
    else:
         print("[FAILURE] Context NOT Established.")

    # 2. Test Follow-up (Invoices)
    print("\n--- 2. Testing Follow-up ('invoice zake') ---")
    query_2 = "Naomba invoice zake"
    response_2 = engine.process_query_v2(query_2)
    print(f"Query: '{query_2}'")
    print(f"Response Preview: {response_2[:100]}...")
    
    if "PASCHAL WHITE" in response_2.upper() and ("INV:" in response_2 or "2026" in response_2):
        print("[SUCCESS] Invoice Context Resolved!")
    else:
        print("[FAILURE] Invoice Context Failed.")

    # 3. Test Another Follow-up (Debt)
    print("\n--- 3. Testing Follow-up ('deni lake') ---")
    query_3 = "Je deni lake ni ngapi?"
    response_3 = engine.process_query_v2(query_3)
    print(f"Query: '{query_3}'")
    print(f"Response: {response_3}")
    
    if "PASCHAL WHITE" in response_3.upper() and ("11,657,000" in response_3):
        print("[SUCCESS] Debt Context Resolved!")
    else:
        print("[FAILURE] Debt Context Failed.")

    # 4. Test Analysis Follow-up
    print("\n--- 4. Testing Follow-up ('mchanganuo wake') ---")
    query_4 = "naomba mchanganuo wake"
    response_4 = engine.process_query_v2(query_4)
    print(f"Query: '{query_4}'")
    # Handle unicode for Windows console
    print(f"Response Preview: {response_4[:100].encode('ascii', 'ignore').decode('ascii')}...")
    
    if "CUSTOMER INTELLIGENCE:" in response_4.upper() and "PASCHAL WHITE" in response_4.upper():
        print("[SUCCESS] Analysis Context Resolved!")
    else:
        print("[FAILURE] Analysis Context Failed.")

    print("\n--- [CONTEXT FLOW VERIFICATION COMPLETE] ---")

if __name__ == "__main__":
    test_context_flow()
