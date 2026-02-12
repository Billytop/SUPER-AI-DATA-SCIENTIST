
import os
import sys
import json

# Set encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from laravel_modules.ai_brain.central_integration_layer import CentralAI

def test_purchase_intelligence():
    print("--- [STARTING PURCHASE INTELLIGENCE VERIFICATION] ---")
    
    ai = CentralAI()
    ai.initialize()
    
    test_cases = [
        {"q": "WHICH PRODUCTS I PURCHASES MOST IN THIS YEAR", "desc": "Top products without false positive greeting"},
        {"q": "NIPE NILINUNUA NINI", "desc": "Purchase list via Swahili slang"},
        {"q": "VITU NILIVONUNUA", "desc": "Purchase list via Swahili keywords"},
        {"q": "LAST YEAR PURCHASES", "desc": "Context setting (Total)"},
        {"q": "GIVE ME LIST", "desc": "Follow-up list request"}
    ]
    
    context = {"connection_id": "DEFAULT"}
    
    for i, test in enumerate(test_cases):
        print(f"\n[{i+1}] Testing: '{test['q']}' ({test['desc']})")
        result = ai.query(test['q'], context)
        
        answer = result.get('answer', '')
        print(f"Response Snippet: {answer[:200]}...")
        
        # Validation
        if "Hi there!" in answer and test['q'] == "WHICH PRODUCTS I PURCHASES MOST IN THIS YEAR":
            print("[FAILURE] Irrelevant greeting still present!")
        elif "LIST YA MANUNUZI" in answer.upper() and ("NILINUNUA" in test['q'] or "LIST" in test['q']):
            print("[SUCCESS] Purchase list correctly identified.")
        elif "Jumla ya manunuzi" in answer and "LAST YEAR" in test['q']:
            print("[SUCCESS] Total purchases correctly identified.")
        else:
            print("[CHECK] Response generated. Verify manually if it fits 'smart' criteria.")

if __name__ == "__main__":
    test_purchase_intelligence()
