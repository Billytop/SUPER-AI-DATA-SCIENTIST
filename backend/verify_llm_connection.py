import sys
import io
import os
import requests
import json

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup Path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Import Config (to test import)
try:
    from config.llm_config import OLLAMA_BASE_URL, DEFAULT_MODEL, SYSTEM_PROMPT
    print("✅ Config Loaded Successfully.")
except ImportError:
    print("❌ Config Import Failed.")
    sys.exit(1)

def test_llm_connection():
    print(f"\n--- TESTING LOCAL LLM CONNECTION ---")
    print(f"URL: {OLLAMA_BASE_URL}")
    print(f"Model: {DEFAULT_MODEL}")
    
    # 1. Check if Ollama is running
    try:
        r = requests.get(OLLAMA_BASE_URL)
        if r.status_code == 200:
            print("✅ Ollama Server is RUNNING.")
        else:
            print(f"⚠️ Ollama Server responded with {r.status_code} (Is it installed?)")
    except Exception as e:
        print(f"❌ Could not connect to Ollama: {e}")
        print("💡 TIP: Make sure you installed Ollama from ollama.com and ran 'ollama serve'")
        return

    # 2. Test Generation
    print("\n--- SENDING TEST PROMPT ---")
    prompt = "Explain briefly what 'OmniBrain' is."
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            res_text = response.json().get("response", "").strip()
            print(f"✅ LLM Response Received:\n'{res_text}'")
        else:
            print(f"❌ Model Generation Failed: {response.text}")
            print("💡 TIP: Did you run 'ollama pull llama3'?")
    except Exception as e:
        print(f"❌ Error during generation: {e}")

if __name__ == "__main__":
    test_llm_connection()
