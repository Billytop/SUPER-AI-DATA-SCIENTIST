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

def verify_live_context():
    print("--- VERIFYING REAL-TIME CONTEXT INJECTION ---")
    
    # Initialize
    try:
        engine = OmnibrainSaaSEngine()
    except Exception as e:
        print(f"❌ Engine Init Failed: {e}")
        return

    # 1. Test _get_live_context Directly
    print("\n1. Fetching Live Stats (SQL Pulse)...")
    start = time.time()
    live_ctx = engine._get_live_context()
    end = time.time()
    
    print(f"⏱️ Time Taken: {end - start:.4f}s")
    print(f"📝 Raw Output:\n---\n{live_ctx}\n---")
    
    # 2. Validation
    if "Live Status" in live_ctx and "Today's Sales" in live_ctx:
        print("✅ SUCCESS: Live Data successfully retrieved from SQL.")
    else:
        print("❌ FAILURE: Live Data missing or malformed.")

    # 3. Simulate RAG + Live Integration
    print("\n2. Simulating Prompt Construction (Dry Run)...")
    query = "How are sales doing right now?"
    rag_results = engine._search_vector_memory(query)
    
    final_prompt = f"User Question: {query}\n"
    if live_ctx:
        final_prompt += f"\n{live_ctx}"
    
    rag_context = ""
    if rag_results:
         rag_context = "\n".join([f"- {r['doc']['output']}" for r in rag_results])
         final_prompt += f"\nRelevant Business Knowledge (History):\n{rag_context}\n"
         
    print(f"📝 Final Prompt to LLM:\n---\n{final_prompt[:500]}...\n---")


if __name__ == "__main__":
    verify_live_context()
