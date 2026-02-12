
import os
import sys

# Ensure the backend path is included
sys.path.append(os.path.join(os.getcwd(), "backend/laravel_modules/ai_brain"))

# Absolute imports for verification
import sovereign_memory_core
import database_unifier
from omnibrain_saas_engine import OmnibrainSaaSEngine

# Fix for Windows Unicode printing
sys.stdout.reconfigure(encoding='utf-8')

def verify_sovereign_memory_grid():
    print("\n--- [STARTING SOVEREIGN MEMORY GRID VERIFICATION (PHASE 36)] ---")
    engine = OmnibrainSaaSEngine()
    
    # 1. Test Memory Core (Persistence)
    print("\n--- 1. Testing Memory Persistence ---")
    query_fact = "Remember that my name is CEO Njuku."
    print(f"User: '{query_fact}'")
    engine.memory_core.remember_interaction(query_fact, "Understood. I have stored this fact.")
    
    query_recall = "What is my name?"
    print(f"User: '{query_recall}'")
    context = engine.memory_core.recall_context(query_recall)
    print(f"System Recall: {context}")

    # 2. Test Database Unifier (Real DB Access)
    print("\n--- 2. Testing Database Unifier ---")
    query_db = "Show me the product Cement"
    print(f"User: '{query_db}'")
    
    # Simulate DB Connection (Mock since we are in a test env without live MySQL credentials sometimes)
    # We will use the Unified Search method which handles the fallback gracefully
    result = engine.db_bridge.unified_search("Cement")
    print(f"DB Bridge Result:\n{result}")

    # 3. Test Unified Processing flow
    print("\n--- 3. Testing Unified Process Flow (V2) ---")
    # This simulates the full pipeline: Memory -> DB -> AI
    final_res = engine.process_query_v2("What is the price of Cement?")
    print(f"Final Engine Response:\n{final_res}")

    print("\n--- [SOVEREIGN MEMORY GRID VERIFICATION COMPLETE] ---")

if __name__ == "__main__":
    verify_sovereign_memory_grid()
