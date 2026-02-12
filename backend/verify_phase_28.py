
import os
import sys
import logging

# Ensure the backend path is included
sys.path.append(os.path.join(os.getcwd(), "backend/laravel_modules/ai_brain"))

import linguistic_core
import statistical_engine
import sovereign_hub
import quantum_ledger
from omnibrain_saas_engine import OmnibrainSaaSEngine

def verify_sovereign_intelligence():
    print("\n--- [STARTING SOVEREIGN VERIFICATION] ---")
    engine = OmnibrainSaaSEngine()
    
    # 1. Test Linguistic Mastery (Sheng/Slang)
    query_sheng = "Nipe mchanganuo wa mshiko na mchongo wa bidhaa bora"
    print(f"\nTesting Linguistics (Sheng): '{query_sheng}'")
    processed = engine.linguistic.process_advanced_request(query_sheng)
    print(f"Result: {processed}")
    
    # 2. Test Strategic Boardroom Debate
    query_strategy = "Nshauri kuhusu strategy ya ujenzi vs maduka ya dawa"
    print(f"\nTesting Boardroom Debate: '{query_strategy}'")
    res_strategy = engine._resolve_business_data(query_strategy)
    print(f"Result Snapshot: {res_strategy[:300]}...")
    
    # 3. Test Forensic Audit (Anomaly Detection)
    print("\nTesting Forensic Audit (Z-Score)...")
    res_audit = engine._run_anomaly_audit_sovereign()
    print(f"Result: {res_audit}")
    
    # 4. Test Treasury Analysis (Liquidity)
    print("\nTesting Treasury (Liquidity Stress Test)...")
    res_treasury = engine._run_treasury_analysis()
    print(f"Result: {res_treasury}")
    
    # 5. Test Strategic Diagnostic
    print("\nTesting 20-Point Strategic Diagnostic...")
    res_diagnostic = engine._run_strategic_diagnostic()
    print(f"Result: {res_diagnostic}")

    print("\n--- [VERIFICATION COMPLETE] ---")

if __name__ == "__main__":
    verify_sovereign_intelligence()
