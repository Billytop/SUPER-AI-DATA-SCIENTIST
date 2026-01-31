
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from laravel_modules.ai_brain.central_integration_layer import CentralAI

def test_query(query):
    print(f"\nQUERY: {query}")
    agent = CentralAI()
    agent.initialize()
    
    context = {
        "connection_id": "TENANT_001",
        "user_id": 1
    }
    
    result = agent.query(query, context)
    print(f"ANSWER:\n{result.get('answer')}")
    print(f"METADATA: {result.get('metadata')}")

if __name__ == "__main__":
    # Test 1: Recent year
    test_query("give me total sales of 2025")
    
    # Test 2: Current year (general)
    test_query("total sales of this year")
    
    # Test 3: Swahili year
    test_query("mauzo ya mwaka 2024")
