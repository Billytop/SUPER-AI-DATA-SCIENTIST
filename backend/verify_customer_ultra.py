import sys
import io
import os

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup Path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Import Engine
from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def test_ultra():
    print("--- TESTING ULTRA CUSTOMER INTELLIGENCE ---")
    engine = OmnibrainSaaSEngine()
    
    # 1. Run Intelligence Report
    print("Query: 'customer info Paschal White'")
    r = engine.process_query_v2("customer info Paschal White")
    
    # 2. Check for New Sections
    checks = {
        "FINANCIAL DNA": "Gross Profit" in r,
        "BEHAVIORAL DNA": "Patterns:" in r,
        "AFFINITY": "Affinity:" in r,
        "OPPORTUNITY": "AI Opportunity:" in r,
        "SUGGESTIONS": "Suggested Next Questions:" in r
    }
    
    all_pass = True
    for section, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{section}: {status}")
        if not passed: all_pass = False
        
    print("\n--- REPORT PREVIEW ---")
    print(r)
    
    if all_pass:
        print("\n🎉 ULTRA UPGRADE SUCCESSFUL!")
    else:
        print("\n⚠️ SOME CHECKS FAILED.")

if __name__ == "__main__":
    test_ultra()
