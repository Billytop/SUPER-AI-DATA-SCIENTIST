from django.conf import settings
from django.core.cache import cache

class AIMemory:
    """
    Manages conversation history for the AI.
    Uses Django Cache (Redis/LocMem) to store the last N exchanges.
    """
    def __init__(self, user_id, session_id='default', window_size=5):
        self.key = f"ai_memory_{user_id}_{session_id}"
        self.window_size = window_size

    def add_exchange(self, user_query, ai_response, sql=None):
        """
        Adds a turn to the history.
        """
        history = self.get_history()
        history.append({
            "user": user_query,
            "ai": ai_response,
            "sql": sql
        })
        
        # Keep only last N turns
        if len(history) > self.window_size:
            history = history[-self.window_size:]
            
        cache.set(self.key, history, timeout=3600) # Expire after 1 hour

    def get_history(self):
        """
        Retrieves current context.
        """
        return cache.get(self.key, [])

    def get_context_string(self):
        """
        Formats history for LLM Prompt.
        """
        history = self.get_history()
        if not history:
            return ""
            
        context = "Previous Conversation:\n"
        for turn in history:
            context += f"User: {turn['user']}\nAI: {turn['ai']}\n"
        return context

    def save_agent_context(self, context_dict):
        """Saves the agent's internal state (filters, intents)"""
        if not context_dict: return
        key = f"{self.key}_state"
        cache.set(key, context_dict, timeout=3600)

    def get_agent_context(self, default=None):
        """Retrieves agent state"""
        key = f"{self.key}_state"
        return cache.get(key, default)
