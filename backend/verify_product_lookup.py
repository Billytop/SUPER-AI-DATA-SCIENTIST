import os
import sys

# Add project root to path
sys.path.append(r'C:\Users\njuku\Documents\AI COMPANY\SephlightyAI\backend')

try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
    brain = OmnibrainSaaSEngine()
    
    # 1. Trigger Product Lookup
    query = 'naomba stock ya holder h/206'
    print(f"Executing query: {query}")
    res = brain.process_query(query, 'test_conn')
    response_str = res.get('response', '')
    
    print(f"AI Response:\n{response_str}")
    
    if "[STOCK INTELLIGENCE]: holder h/206" in response_str or "H/206" in response_str:
        print("SUCCESS: Product stock correctly resolved via SQL Bridge!")
        if "Stock Started" in response_str and "Current Balance" in response_str:
            print("SUCCESS: Smart Flow data detected!")
            sys.exit(0)
        else:
            print("ERROR: Smart Flow data missing.")
            sys.exit(1)
    else:
        print("ERROR: Product lookup failed or returned generic response.")
        sys.exit(2)

except Exception as e:
    print(f"Error: {e}")
    sys.exit(5)
