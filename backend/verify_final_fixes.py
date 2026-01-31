
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Force UTF-8 output for Windows terminals
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from laravel_modules.ai_brain.central_integration_layer import CentralAI

def test_final(query):
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
    # Test 1: Identity (New variant)
    test_final("who are u")
    
    # Test 2: Sales 2025
    test_final("total sales of 2025")
    
    # Test 3: Expenses 2025 (The failing one)
    test_final("expenses 2025")
    
    # Test 4: Swahili variant for expenses
    test_final("matumizi ya mwaka 2025")
