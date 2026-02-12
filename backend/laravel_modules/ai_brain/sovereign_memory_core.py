
import json
import os
import datetime
from pathlib import Path
from typing import Dict, Any, List

# SOVEREIGN MEMORY CORE v1.0
# Persistent Memory Store for User Context and Preferences.

class SovereignMemory:
    def __init__(self):
        self.memory_path = Path(__file__).parent.parent.parent / "data" / "sovereign_memory.json"
        self.short_term_buffer = [] # Last 10 interactions
        self.long_term_store = self._load_memory()

    def remember_interaction(self, query: str, response: str) -> None:
        """
        Stores a query/response pair in short-term buffer and potentially long-term.
        """
        interaction = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "response": response
        }
        
        self.short_term_buffer.append(interaction)
        if len(self.short_term_buffer) > 10:
            self.short_term_buffer.pop(0)

        # Naive "Important" Check for Long-Term Storage
        if any(w in query.lower() for w in ["my name is", "i prefer", "always use", "remember that"]):
            self._store_long_term_fact(query)

    def recall_context(self, current_query: str) -> str:
        """
        Retrieves relevant past information based on keywords.
        """
        # 1. Check Long-Term Facts (Preferences)
        facts = self.long_term_store.get("facts", [])
        relevant_facts = [f for f in facts if any(token in f.lower() for token in current_query.split())]
        
        # 2. Check Recent Context
        recent = ""
        if self.short_term_buffer:
            last = self.short_term_buffer[-1]
            recent = f"Last Topic: {last['query']}"
            
        context_str = ""
        if relevant_facts:
            context_str += f"[RECALLED FACTS]: {'; '.join(relevant_facts[:3])}\n"
        if recent:
            context_str += f"[RECENT CONTEXT]: {recent}"
            
        return context_str

    def _store_long_term_fact(self, text: str) -> None:
        """Saves a specific fact to persistent storage."""
        if "facts" not in self.long_term_store:
            self.long_term_store["facts"] = []
            
        self.long_term_store["facts"].append(text)
        self._save_memory()

    def _load_memory(self) -> Dict[str, Any]:
        """Loads JSON memory file."""
        if not self.memory_path.exists():
            return {}
        try:
            with open(self.memory_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_memory(self) -> None:
        """Persists memory to disk."""
        # Ensure directory exists
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, "w", encoding='utf-8') as f:
            json.dump(self.long_term_store, f, indent=2)

MEMORY_CORE = SovereignMemory()
