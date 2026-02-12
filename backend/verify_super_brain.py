"""
Verification Script: Super Brain Contextual Sovereignty (Phase 41/42)
Tests:
1. Domain Locking (Purchases -> "last year" should stay in purchases)
2. Granularity Sovereignty (List mode -> "this month" should stay in list mode)
3. Hyper-Scale Suggestions (Specific strategic leads from Ultimate Matrix)
"""

import sys
import os
import json

import sys
import os
import json

# Correct Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    # Use full package path
    from laravel_modules.ai_brain.central_integration_layer import CentralAI
    print("SUCCESS: Integration Layer Imported via Package Path.")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

def test_super_brain():
    ai = CentralAI()
    conn_id = "2026newv1"
    
    print("\n--- TEST 1: Purchase Domain Locking ---")
    q1 = "Nipe manunuzi yangu ya jana"
    print(f"Query 1: {q1}")
    res1 = ai.query(q1, {'connection_id': conn_id})
    # Sanitize output for printing (Windows console safety)
    safe_ans1 = str(res1['answer']).encode('ascii', 'ignore').decode('ascii')
    print(f"Response 1: {safe_ans1[:150]}...")
    
    q2 = "na ya mwaka jana?"
    print(f"Query 2: {q2}")
    res2 = ai.query(q2, {'connection_id': conn_id})
    safe_ans2 = str(res2['answer']).encode('ascii', 'ignore').decode('ascii')
    print(f"Response 2: {safe_ans2[:150]}...")
    
    if "manunuzi" in res2['answer'].lower() or "purchase" in res2['answer'].lower():
        print("OK: Domain Locking maintained (Stayed in Purchases).")
    else:
        print("FAIL: Domain Mixup detected.")

    print("\n--- TEST 2: Granularity Sovereignty (Un-Rushed Listing) ---")
    q3 = "Orodha ya matumizi ya mwezi huu"
    print(f"Query 3: {q3}")
    res3 = ai.query(q3, {'connection_id': conn_id})
    safe_ans3 = str(res3['answer']).encode('ascii', 'ignore').decode('ascii')
    print(f"Response 3: {safe_ans3[:150]}...")
    
    q4 = "na mwezi uliopita?" 
    print(f"Query 4: {q4}")
    res4 = ai.query(q4, {'connection_id': conn_id})
    safe_ans4 = str(res4['answer']).encode('ascii', 'ignore').decode('ascii')
    print(f"Response 4: {safe_ans4[:150]}...")
    
    if "INV:" in res4['answer'] or "list" in res4['answer'].lower() or "vitu" in res4['answer'].lower():
        print("OK: Granularity Sovereignty maintained (Stayed in List Mode).")
    else:
        print("FAIL: Rushed to total (Granularity lost).")

    print("\n--- TEST 3: Hyper-Scale Heuristics (Phase 42) ---")
    # Check both answer text and metadata suggestions safely
    suggestions = res2.get('metadata', {}).get('suggestions', [])
    safe_suggestions = [str(s).encode('ascii', 'ignore').decode('ascii') for s in suggestions]
    
    has_strategic = "STRATEGIC LEAD" in safe_ans2 or "LEAKAGE AUDIT" in safe_ans4 or any("STRATEGIC LEAD" in s for s in safe_suggestions)
    
    if has_strategic:
         print("OK: Hyper-Scale Logic Matrix providing specific strategic leads.")
    else:
         print("WARNING: Strategic leads not detected in response text.")

if __name__ == "__main__":
    test_super_brain()
