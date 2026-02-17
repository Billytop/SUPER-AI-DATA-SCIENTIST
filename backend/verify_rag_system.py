import sys
import io
import os
import time

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup Path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Import Engine
try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
except ImportError as e:
    print(f"❌ Could not import Engine: {e}")
    sys.exit(1)

def test_rag():
    print("\n--- TESTING RAG (VECTOR MEMORY) ---")
    
    print("🧠 Initializing Engine (Loading Vectors)...")
    start = time.time()
    engine = OmnibrainSaaSEngine()
    end = time.time()
    print(f"✅ Engine Loaded in {end - start:.2f}s")
    
    # 1. Check Index Status
    if engine.vector_index is not None:
        print(f"✅ Vector Index Loaded: {len(engine.rag_docs)} documents.")
    else:
        print("❌ Vector Index FAILED to load.")
        return

    # 2. Test Search
    queries = [
        "How is Paschal's debt?",
        "What is the SKU for 13A Plug?",
        "Explain OmniBrain internal logic"
    ]
    
    for q in queries:
        print(f"\n🔍 Searching for: '{q}'")
        results = engine._search_vector_memory(q, top_k=1)
        if results:
            print(f"✅ Found Match (Score: {results[0]['score']:.2f}):")
            print(f"   📄 {results[0]['doc']['output'][:100]}...")
        else:
            print("⚠️ No relevant context found.")

if __name__ == "__main__":
    test_rag()
