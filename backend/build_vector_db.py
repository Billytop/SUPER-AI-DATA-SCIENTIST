import json
import pickle
import os
import sys
import io

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("❌ Critical Library Missing: sentence-transformers")
    print("Run: pip install sentence-transformers")
    sys.exit(1)

def build_index():
    print("--- BUILDING VECTOR MEMORY FOR SEPHLIGHTY AI ---")
    
    # 1. Load Data
    data_file = "sephlighty_training_data.jsonl"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    documents = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
            
    print(f"📚 Loaded {len(documents)} knowledge items.")

    # 2. Load Embedding Model (optimised for CPU/Low-VRAM)
    print("🧠 Loading Embedding Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2') 

    # 3. Create Embeddings
    print("🔢 Converting text to vectors...")
    # We embed the "instruction" + "input" (The Question/Context)
    texts = [doc["instruction"] + " " + doc.get("input", "") for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)

    # 4. Save Index and Data
    # For a small dataset (<100k items), a simple pickle of arrays is faster/easier than FAISS
    index_data = {
        "embeddings": embeddings,
        "documents": documents
    }
    
    with open("vector_memory.pkl", "wb") as f:
        pickle.dump(index_data, f)
        
    print(f"✅ SUCCESS: Vector Memory built saved to 'vector_memory.pkl'")
    print("🚀 OmniBrain can now recall this information instantly.")

if __name__ == "__main__":
    import io
    build_index()
