
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Force UTF-8 output for Windows terminals
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from laravel_modules.ai_brain.central_integration_layer import CentralAI

def test_complex_logic(queries):
    agent = CentralAI()
    agent.initialize()
    
    context = {
        "connection_id": "TENANT_001",
        "user_id": 1
    }
    
    for query in queries:
        print(f"\nQUERY: {query}")
        result = agent.query(query, context)
        print(f"ANSWER:\n{result.get('answer')}")

if __name__ == "__main__":
    print("--- COMPLEX COMPARISON TEST ---")
    test_complex_logic([
        "NIPE EXPENSES ZA MWAKAHUU",
        "ZA MWAKA JANA",
        "COMPARE HIZO MBILEE",
        "NIIPI INA EXPENSES KUBWA ZAIDI"
    ])
