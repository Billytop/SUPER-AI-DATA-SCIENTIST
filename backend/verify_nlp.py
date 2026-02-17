import sys
import io
import os
import logging

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup Path
backend_dir = os.getcwd()
sys.path.append(backend_dir)

# Import Engine
try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
except ImportError as e:
    print(f"❌ Could not import Engine: {e}")
    sys.exit(1)

# Configure Logging (suppress heavy debug)
logging.basicConfig(level=logging.ERROR)

def verify_nlp():
    print("--- 🗣️ VERIFYING TRUE NLP (THE VOICE) ---")
    
    try:
        engine = OmnibrainSaaSEngine()
        print("Omnibrain connected.")
        
        queries = [
            "How many users are there?",
            "What is the total sales for today?"
        ]
        
        for q in queries:
            print(f"\nQuestion: '{q}'")
            response = engine.process_query_v2(q)
            print(f"AI Response: {response}")
            
            # Simple heuristic check
            if len(response) > 50 and " " in response:
                print("✅ Likely NLP (Long sentence)")
            else:
                print("⚠️ Likely Raw (Short result)")
        
    except Exception as e:
        print(f"❌ Verification Failed: {e}")

if __name__ == "__main__":
    verify_nlp()
