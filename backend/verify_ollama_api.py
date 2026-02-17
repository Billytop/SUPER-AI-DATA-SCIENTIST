import requests
import sys
import io

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_ollama():
    print("--- 🧠 VERIFYING OLLAMA API ---")
    try:
        # Check if running
        resp = requests.get('http://localhost:11434/')
        if resp.status_code == 200:
            print("✅ Ollama Service is RUNNING.")
        else:
            print(f"❌ Ollama Service responded with {resp.status_code}")
            return
            
        # Check Models
        resp = requests.get('http://localhost:11434/api/tags')
        if resp.status_code == 200:
            models = resp.json().get('models', [])
            model_names = [m['name'] for m in models]
            print(f"📚 Installed Models: {model_names}")
            
            if any('llama3.2' in m for m in model_names) or any('phi3' in m for m in model_names):
                print("✅ Compatible model found!")
            else:
                print("⚠️ No standard model found. User might need to run 'ollama run llama3.2:1b'")
        
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("Please ensure Ollama is running (Icon in system tray).")

if __name__ == "__main__":
    check_ollama()
