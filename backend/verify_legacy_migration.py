
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Force UTF-8 output for Windows terminals
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from laravel_modules.ai_brain.central_integration_layer import CentralAI

def test_knowledge(query):
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
    # Test 1: Identity (Swahili)
    test_knowledge("wewe ni nani")
    
    # Test 2: Definition (EBITDA)
    test_knowledge("what is ebitda")
    
    # Test 3: Tax rules (Swahili)
    test_knowledge("sheria za kodi vat")
    
    # Test 4: Swahili typo normalization
    test_knowledge("nshauri mauzo ya mwaka jana")
