
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Force UTF-8 output for Windows terminals
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from laravel_modules.ai_brain.central_integration_layer import CentralAI

def test_chart_followup():
    agent = CentralAI()
    agent.initialize()
    
    context = {
        "connection_id": "TENANT_001",
        "user_id": 1
    }
    
    # Simulate a follow-up query expanded by cognitive context
    # Usually the frontend sends just "chart", and Cognitive adds context
    # We will simulate the "expanded" query that OmniBrain receives
    queries = [
        "chart (kuhusiana na: i want to know my total sales this year)"
    ]
    
    for query in queries:
        print(f"\nQUERY: {query}")
        result = agent.query(query, context)
        print(f"ANSWER:\n{result.get('answer')}")
        if "[CHART_DATA]" in result.get('answer', ''):
             print("\n✅ CHART_DATA DETECTED AT TOP!")
        else:
             print("\n❌ CHART_DATA MISSING OR BURIED!")

if __name__ == "__main__":
    print("--- CHART FOLLOW-UP TEST ---")
    test_chart_followup()
