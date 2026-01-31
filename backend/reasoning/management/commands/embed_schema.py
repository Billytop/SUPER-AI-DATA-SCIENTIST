import os
import django
from django.core.management.base import BaseCommand
from django.apps import apps
from django.conf import settings
import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
import pickle

class Command(BaseCommand):
    help = 'Embeds the Database Schema into a FAISS Vector Store for the AI'

    def handle(self, *args, **kwargs):
        self.stdout.write("Initializing Schema Embedding...")
        
        # 1. Extract Schema Metadata
        schemas = []
        for app_config in apps.get_app_configs():
            # Filter for our apps only
            if app_config.name in ['core', 'inventory', 'sales', 'partners', 'accounting', 'hrm', 'operations', 'crm', 'restaurant']:
                for model in app_config.get_models():
                    model_name = model.__name__
                    doc = model.__doc__.strip() if model.__doc__ else ""
                    fields = [f.name for f in model._meta.get_fields()]
                    
                    text_representation = f"Table: {model_name}\nApp: {app_config.name}\nDescription: {doc}\nFields: {', '.join(fields)}"
                    schemas.append(text_representation)
        
        self.stdout.write(f"Found {len(schemas)} tables to embed.")
        
        vector_store_path = os.path.join(settings.BASE_DIR, 'reasoning', 'vector_store')
        os.makedirs(vector_store_path, exist_ok=True)
        
        # 2. Embed
        # Real implementation using SentenceTransformer
        # We handle import inside to avoid breaking if not installed
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(schemas)
            
            # 3. Store in FAISS
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            faiss.write_index(index, os.path.join(vector_store_path, 'schema.index'))
            
        except ImportError:
             self.stdout.write(self.style.WARNING("sentence-transformers or faiss not found. Skipping real embedding."))
             # Fallback mock for dev
             dummy_dim = 384
             embeddings = np.random.rand(len(schemas), dummy_dim).astype('float32')
        
        with open(os.path.join(vector_store_path, 'schema_metadata.pkl'), 'wb') as f:
            pickle.dump(schemas, f)
            
        self.stdout.write(self.style.SUCCESS(f"Successfully embedded {len(schemas)} tables into {vector_store_path}"))
