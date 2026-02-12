
import os
import sys
import logging

# Ensure the backend path is included
sys.path.append(os.path.join(os.getcwd(), "backend/laravel_modules/ai_brain"))

# Absolute imports for verification
import sovereign_tax_code_global
import global_trade_routes_db
import medical_diagnostic_db
from omnibrain_saas_engine import OmnibrainSaaSEngine

# Fix for Windows Unicode printing
sys.stdout.reconfigure(encoding='utf-8')

def verify_universal_knowledge():
    print("\n--- [STARTING UNIVERSAL KNOWLEDGE VERIFICATION (PHASE 31)] ---")
    engine = OmnibrainSaaSEngine()
    
    # 1. Test Global Tax Engine
    print("\n--- 1. Testing Sovereign Global Tax Engine ---")
    query_tax = "Tax rate in Angola (Corporate Tax)"
    print(f"Query: '{query_tax}'")
    res_tax = engine._run_global_tax_query(query_tax)
    print(f"Result:\n{res_tax}")
    
    query_tax_2 = "What is the VAT in Belgium?"
    print(f"Query: '{query_tax_2}'")
    res_tax_2 = engine._run_global_tax_query(query_tax_2)
    print(f"\nResult:\n{res_tax_2}")

    # 2. Test Global Logistics DB
    print("\n--- 2. Testing Global Trade Routes ---")
    query_logistics = "Calculate distance to Dubai"
    print(f"Query: '{query_logistics}'")
    res_logistics = engine._run_global_logistics(query_logistics)
    print(f"Result:\n{res_logistics}")
    
    query_shipping = "Estimate shipping cost to Shanghai"
    print(f"Query: '{query_shipping}'")
    res_shipping = engine._run_global_logistics(query_shipping)
    print(f"Result:\n{res_shipping}")

    # 3. Test Medical Diagnostic Engine
    print("\n--- 3. Testing Medical Diagnostic Engine ---")
    query_med = "I have a severe headache"
    print(f"Query: '{query_med}'")
    # Simulate routing logic manually if regex doesn't catch exact phrase
    res_med = engine.medical_db.diagnose(query_med) 
    print(f"Result:\n{res_med}")
    
    query_med_2 = "Symptoms of acidity"
    print(f"\nQuery: '{query_med_2}'")
    res_med_2 = engine.medical_db.diagnose(query_med_2)
    print(f"Result:\n{res_med_2}")

    print("\n--- [UNIVERSAL KNOWLEDGE VERIFICATION COMPLETE] ---")

if __name__ == "__main__":
    verify_universal_knowledge()
