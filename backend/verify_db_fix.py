
import os
import django
import sys

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from reasoning.agents import SephlightyBrain

def verify():
    print("Initializing Agent...")
    try:
        agent = SephlightyBrain()
        
        queries = [
            "shw bst product",
            "by amount",
            "best products by revenue"
        ]
        
        for q in queries:
            print(f"\n--- Testing: '{q}' ---")
            clean_q, _ = agent.preprocess(q)
            print(f"CLEAN: {clean_q}")
            response = agent.run(q)
            print(f"INTENT: {response.get('intent')}")
            if "System Error" in str(response.get('answer')):
                print(f"FAILED: {response.get('answer')}")
            else:
                print(f"SUCCESS. Answer snippet: {str(response.get('answer'))[:100]}...")
        
        print(f"Response keys: {response.keys()}")
        
        if response.get('answer') and "System Error" not in response['answer']:
            print("SUCCESS: Database query executed successfully.")
            print(f"Answer: {response['answer']}")
        else:
            print("FAILURE: System Error still present.")
            print(f"Answer: {response.get('answer')}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    verify()
