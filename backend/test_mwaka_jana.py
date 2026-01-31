import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from reasoning.agents import SQLReasoningAgent

agent = SQLReasoningAgent()

# The exact query the user tested
query = "mfanyakazi bora ni nani kwa mauzo mwaka jana"

print(f"Testing: {query}")
print("="*80)

result = agent.run(query)

print(f"Answer: {result['answer']}")
print(f"\nGenerated SQL:")
print(result['sql'])

# Verify it has the correct filter
if "YEAR(transaction_date) = YEAR(CURDATE()) - 1" in result['sql']:
    print("\n[SUCCESS] Correctly detected 'mwaka jana' as LAST YEAR filter!")
elif "CURDATE() - INTERVAL 1 DAY" in result['sql']:
    print("\n[FAIL] Incorrectly detected as YESTERDAY filter")
else:
    print("\n[UNKNOWN] Unexpected SQL filter")
