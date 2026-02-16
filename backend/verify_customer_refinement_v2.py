import os
import sys

# Setup Path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Import Engine
from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def test_v2():
    engine = OmnibrainSaaSEngine()
    
    # Test 1: Info
    r1 = engine.process_query_v2("customer info Paschal White")
    print(f"TEST 1 (INFO): {'PASS' if 'Super Advanced Intelligence' in r1 else 'FAIL'}")
    
    # Test 2: Debt
    r2 = engine.process_query_v2("deni la Paschal White")
    print(f"TEST 2 (DEBT): {'PASS' if 'Deni la' in r2 else 'FAIL'}")
    
    # Test 3: Range
    r3 = engine.process_query_v2("list orders from 2026-01-01 to 2026-02-01")
    print(f"TEST 3 (RANGE): {'PASS' if 'LIST YA' in r3 or 'Jumla ya' in r3 else 'FAIL'}")
    
    # Final Confirmation print
    print("---ALL TESTS COMPLETED---")

if __name__ == "__main__":
    test_v2()
