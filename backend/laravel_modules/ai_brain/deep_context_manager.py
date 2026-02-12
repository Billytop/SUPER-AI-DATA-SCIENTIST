
import json
import os
import time
from typing import Dict, Any

class DeepContextManager:
    """
    Sovereign Memory Core: Weighted Context Logic.
    Decides what to remember and what to forget based on importance.
    """
    def __init__(self, context_file=None):
        if context_file is None:
            # Use absolute path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.context_file = os.path.join(base_dir, "context_store.json")
        else:
            self.context_file = context_file
            
        self.context = self._load_context()
        
    def _load_context(self) -> Dict[str, Any]:
        if os.path.exists(self.context_file):
            try:
                with open(self.context_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_context(self):
        try:
            with open(self.context_file, 'w') as f:
                json.dump(self.context, f)
        except Exception as e:
            print(f"Context Save Error: {e}")

    def add_context_item(self, key: str, value: Any, weight: int = 1):
        """
        Adds an item to memory with a weight.
        High weight (e.g. 10) = Keep longer.
        Low weight (e.g. 1) = Discard soon.
        """
        self.context[key] = {
            "value": value,
            "weight": weight,
            "timestamp": time.time()
        }
        self._prune_memory()
        self._save_context()

    def get_context_item(self, key: str) -> Any:
        item = self.context.get(key)
        if item:
            # Refresh timestamp on access
            item["timestamp"] = time.time()
            self._save_context()
            return item["value"]
        return None

    def _prune_memory(self):
        """
        Removes old, low-weight items.
        Logic: Use weight as 'hours to live'.
        """
        now = time.time()
        keys_to_remove = []
        
        for k, v in self.context.items():
            age_hours = (now - v["timestamp"]) / 3600
            # If age exceeds weight (hours), forget it
            if age_hours > v["weight"]:
                keys_to_remove.append(k)
                
        for k in keys_to_remove:
            del self.context[k]

    def reset_short_term(self):
        """Clears only low-weight items (conversation flow), keeps high-weight (facts)."""
        keys_to_remove = [k for k, v in self.context.items() if v["weight"] < 5]
        for k in keys_to_remove:
            del self.context[k]
        self._save_context()
