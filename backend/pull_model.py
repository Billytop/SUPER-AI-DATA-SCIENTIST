import requests
import json
import sys
import io

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def pull_llama():
    print("--- ⬇️ STARTING MODEL DOWNLOAD (llama3.2:1b) ---")
    url = "http://localhost:11434/api/pull"
    payload = {"name": "llama3.2:1b", "stream": False}
    
    try:
        print("Sending pull request to Ollama... (This may take a minute)")
        resp = requests.post(url, json=payload)
        
        if resp.status_code == 200:
            print("✅ Model Pull Successful!")
            print(resp.text)
        else:
            print(f"❌ Pull Failed: {resp.status_code} - {resp.text}")
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    pull_llama()
