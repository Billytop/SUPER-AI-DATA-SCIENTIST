import logging
import traceback
# Mock logging
logging.basicConfig(level=logging.INFO)

from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

engine = OmnibrainSaaSEngine()

queries = [
    "Identify top 5 selling products",
    "best selling category this year",
    "Identify top 5 selling products this year",
    "best selling category last year",
    "best sales by staff this year",
    "bora 5 bidhaa mwaka huu"
]

print("\n--- Testing Upgraded Analytics Engine ---")
for q in queries:
    print(f"\nQuery: {q}")
    try:
        res = engine.process_query(q, "DEFAULT")
        print(f"Response:\n{res['response']}")
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
print("\n--- Testing End ---")
