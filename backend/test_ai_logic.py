import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from reasoning.agents import SQLReasoningAgent
from reasoning.nlp import NaturalLanguageProcessor

# Test the problematic query
test_query = "mfanyakazi bora ni nani kwa mauzo"

nlp = NaturalLanguageProcessor()
agent = SQLReasoningAgent()

print(f"Original Query: {test_query}")
print(f"Query Lower: {test_query.lower()}")

# Test NLP processing
clean_q = nlp.clean_query(test_query)
print(f"Cleaned Query: '{clean_q}'")

# Check conditions
q_lower = test_query.lower()
print(f"\nCondition Checks:")
print(f"  'employee' in clean_q: {'employee' in clean_q}")
print(f"  'mfanyakazi' in q_lower: {'mfanyakazi' in q_lower}")
print(f"  'best' in clean_q: {'best' in clean_q}")
print(f"  'top' in clean_q: {'top' in clean_q}")
print(f"  'sales' in clean_q: {'sales' in clean_q}")
print(f"  'sale' in clean_q: {'sale' in clean_q}")
print(f"  'mauzo' in q_lower: {'mauzo' in q_lower}")

# Test full agent
print(f"\n{'='*60}")
print("Running Full Agent:")
print(f"{'='*60}")
result = agent.run(test_query)
print(f"Answer: {result['answer']}")
print(f"SQL: {result['sql']}")
