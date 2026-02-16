import sys
import io
import os

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup Path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Import Engine
from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def test_debt():
    print("--- TESTING DEBT INTENT ---")
    engine = OmnibrainSaaSEngine()
    
    queries = [
        ("GENERAL DEBT", "Who owes me money?"),
        ("SPECIFIC DEBT", "Debt for Paschal White")
    ]
    
    all_passed = True
    for label, q in queries:
        res = engine.process_query_v2(q)
        passed = False
        
        if label == "GENERAL DEBT":
            # Check for title or list
            if "Outstanding Customer Debts" in res or "No outstanding debts" in res:
                passed = True
        elif label == "SPECIFIC DEBT":
             if "Deni la **Paschal" in res or "hana deni" in res:
                 passed = True
                 
        status = "PASS" if passed else "FAIL"
        print(f"{label}: {status}")
        if not passed:
            print(f"FAILED RESPONSE: {res}")
            all_passed = False
            
    if all_passed:
        print("\nALL DEBT TESTS PASSED.")
    else:
        print("\nSOME DEBT TESTS FAILED.")

if __name__ == "__main__":
    test_debt()
