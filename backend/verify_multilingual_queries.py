
import os
import sys

# Set encoding for Windows console to handle UTF-8/Chinese
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from laravel_modules.ai_brain.central_integration_layer import CentralAI

def test_chinese_query():
    print("--- [STARTING MULTILINGUAL QUERY VERIFICATION] ---")
    
    ai = CentralAI()
    ai.initialize()
    
    # Test Case: Chinese query for 2026 sales
    query = "2026年总销售额"
    context = {"connection_id": "TENANT_001"}
    
    print(f"\nTesting Query: '{query}'")
    print(f"Tenant ID: {context['connection_id']}")
    
    result = ai.query(query, context)
    
    # In a real environment, it might fail SQL execution if DB is not fully mocked, 
    # but we want to see if it reaches the SaaS engine with high confidence.
    
    print(f"\nResponse: {result.get('answer', 'No answer')}")
    print(f"Intent: {result.get('intent', 'No intent')}")
    
    metadata = result.get('metadata', {})
    confidence = metadata.get('confidence', 0)
    reasoning = metadata.get('reasoning', 'No reasoning')
    
    print(f"Confidence: {confidence}")
    print(f"Reasoning: {reasoning}")
    
    if confidence >= 0.7:
        print(f"[SUCCESS] High confidence achieved for Chinese query.")
    else:
        print(f"[FAILURE] Low confidence: {reasoning}")

if __name__ == "__main__":
    test_chinese_query()
