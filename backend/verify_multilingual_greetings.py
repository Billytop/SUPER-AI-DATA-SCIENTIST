
import os
import sys

# Set encoding for Windows console to handle UTF-8/Chinese
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from laravel_modules.ai_brain.cognitive_intelligence_ai import CognitiveIntelligenceAI

def test_multilingual_greetings():
    print("--- [STARTING MULTILINGUAL GREETINGS VERIFICATION] ---")
    
    cog = CognitiveIntelligenceAI()
    
    test_cases = [
        ("bonju", "Bonjour"),
        ("嘿", "你好"),
        ("hola", "Hola"),
        ("ciao", "Ciao"),
        ("namaste", "Namaste"),
        ("jambo", "Jambo"),
        ("hi", "Hi there"),
        ("habari", "Habari yako")
    ]
    
    passed = 0
    for query, expected_snippet in test_cases:
        print(f"\nTesting Query: '{query}'")
        response = cog.process_query(query)['response']
        print(f"Response: {response}")
        
        # Check if robotic reasoning is present
        robotic_text = "Executing first-principles breakdown"
        if robotic_text in response:
            print(f"[FAILURE] Robotic reasoning detected for '{query}'")
        elif expected_snippet.lower() in response.lower():
            print(f"[SUCCESS] Natural greeting response found.")
            passed += 1
        else:
            print(f"[FAILURE] Dynamic greeting not found.")

    print(f"\n--- [VERIFICATION COMPLETE: {passed}/{len(test_cases)} PASSED] ---")

if __name__ == "__main__":
    test_multilingual_greetings()
