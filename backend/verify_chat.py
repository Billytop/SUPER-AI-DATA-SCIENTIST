import sys
import io
import os
import logging
import time

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

# Configure Logging
logging.basicConfig(level=logging.ERROR)

def verify_chat():
    print("--- 💬 VERIFYING PURE CHAT & PERSONA ---")
    
    try:
        engine = OmnibrainSaaSEngine()
        print("Omnibrain connected.")
        
        queries = [
            "Hello there!",
            "Who are you?",
            "Tell me a short joke about business."
        ]
        
        for q in queries:
            print(f"\nUser: '{q}'")
            start = time.time()
            response = engine.process_query_v2(q)
            duration = time.time() - start
            
            print(f"AI ({duration:.1f}s): {response}")
            
            if response and len(response) > 5:
                print("✅ Valid Response")
            else:
                print("❌ No Response (Failed)")
        
    except Exception as e:
        print(f"❌ Verification Failed: {e}")

if __name__ == "__main__":
    verify_chat()
