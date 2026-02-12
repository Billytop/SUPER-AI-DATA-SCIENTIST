import os
import sys

# Add project root to path
sys.path.append(r'C:\Users\njuku\Documents\AI COMPANY\SephlightyAI\backend')

try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
    brain = OmnibrainSaaSEngine()
    
    q = "draw gaph of sales of this year"
    q_adv = brain.linguistic.process_advanced_request(q)
    cleaned = brain._clean_query(q_adv)
    
    pronouns = ["zake", "yake", "lake", "wake", "wao", "huyo", "his", "her", "that customer", "him", "yangu", "kwake"]
    matches = [p for p in pronouns if p in cleaned]
    
    print(f"Original: '{q}'")
    print(f"Advanced: '{q_adv}'")
    print(f"Cleaned: '{cleaned}'")
    print(f"Pronoun Matches: {matches}")
    
    # Check if 'his' is in 'graph graph sales this year'? No.
    # What if 'this' maps to something?
    
except Exception as e:
    print(f"Error: {e}")
