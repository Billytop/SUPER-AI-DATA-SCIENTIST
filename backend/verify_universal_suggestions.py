import sys
import io
import os

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup Path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Import Engine
from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def test_suggestions():
    print("--- TESTING UNIVERSAL SUGGESTIONS ---")
    engine = OmnibrainSaaSEngine()
    
    queries = [
        ("SALES", "sales this month"),
        ("PURCHASE", "show purchases for this year"),
        ("CUSTOMER", "customer info Paschal White"),
        ("HABIT", "When does Paschal White usually shop?"),
        ("PAYMENT", "Show payment history for Paschal White")
    ]
    
    print("\n--- SUMMARY ---")
    all_passed = True
    for label, q in queries:
        res = engine.process_query_v2(q)
        passed = False
        
        if label == "HABIT":
            if "Habit Analysis" in res: passed = True
        elif label == "PAYMENT":
            if "Payment History" in res: passed = True
        else:
            if "Suggested Next Questions" in res: passed = True
            
        status = "PASS" if passed else "FAIL"
        print(f"{label}: {status}")
        if not passed: 
            print(f"FAILED RESPONSE: {res}")
            all_passed = False
        
    if all_passed:
        print("\nALL TESTS PASSED.")
    else:
        print("\nSOME TESTS FAILED.")

if __name__ == "__main__":
    test_suggestions()
