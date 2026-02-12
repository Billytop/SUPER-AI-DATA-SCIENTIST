import os
import sys

# Add project root to path
sys.path.append(r'C:\Users\njuku\Documents\AI COMPANY\SephlightyAI\backend')

try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
    brain = OmnibrainSaaSEngine()
    
    queries = [
        "last",
        "last year",
        "mwaka jana",
        "mwak jana",
        "draw gaph of sales"
    ]
    
    for q in queries:
        cleaned = brain._clean_query(q)
        print(f"Original: '{q}' -> Cleaned: '{cleaned}'")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
