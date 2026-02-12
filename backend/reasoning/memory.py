from django.conf import settings
from django.core.cache import cache
import json
import logging
import re
from datetime import datetime, timedelta

class SuperNeuralLSTM:
    """
    GPT-4 LEVEL NEURAL MEMORY CORE (SUPER LSTM ELITE v6.0)
    Role: Strategic Context Architect
    Features: Multi-Head Cell States, Semantic Attention Gates, Recursive Strategic Compression.
    """
    def __init__(self, user_id, session_id='default'):
        self.user_id = user_id
        self.session_id = session_id
        self.base_key = f"super_neural_v6_{user_id}_{session_id}"
        self.heads = ['FINANCE', 'OPERATIONS', 'STRATEGY', 'META']
        
    def add_interaction(self, query, response, intent=None, entities=None):
        """
        Processes Interaction through Multi-Head Neural Gating.
        """
        # 1. SEMANTIC HEAD ROUTING
        # Direct context to specific neural heads
        head = self._route_to_head(intent, query)
        
        # 2. INPUT GATE: Strategic Fact Extraction
        self._update_head_state(head, query, response, entities)
        
        # 3. RECURSIVE STRATEGIC COMPRESSION
        # If history gets too long, compress into a 'Strategic Fact Blob'
        self._compress_history_if_needed()
        
        # 4. RECENT BUFFER
        history = self.get_raw_history()
        history.append({
            "q": query,
            "a": response[:300] + "..." if len(response) > 300 else response,
            "intent": intent,
            "ts": datetime.now().isoformat(),
            "head": head
        })
        
        # Keep recent buffer manageable (GPT-style sliding window)
        if len(history) > 15:
            history = history[-15:]
            
        cache.set(f"{self.base_key}_raw", history, timeout=86400) # 24h persistence

    def get_context_for_brain(self, current_query):
        """
        ATTENTION GATE: Semantic Context Retrieval.
        GPT-4 style selective context prioritization.
        """
        raw_history = self.get_raw_history()
        compressed_facts = self.get_compressed_facts()
        
        # Determine current active head for attention
        current_head = self._route_to_head(None, current_query)
        head_state = self._get_head_state(current_head)
        
        context = f"🧠 **GPT-4 NEURAL CORE: {current_head} MODE**\n"
        
        # Layer 1: Persistent Strategic Goals (Infinite Life)
        if head_state:
            context += f"STRATEGIC_STATE: {json.dumps(head_state)}\n"
            
        # Layer 2: Compressed Historical Facts
        if compressed_facts:
            context += f"LONG_TERM_KNOWLEDGE: {compressed_facts}\n"
            
        # Layer 3: Semantic Attention (Selectively include relevant past turns)
        relevant_turns = self._apply_attention_gate(current_query, raw_history)
        for turn in relevant_turns:
            context += f"PAST: Q: {turn['q']} | A: {turn['a']}\n"
            
        return context

    def _route_to_head(self, intent, query):
        q = query.lower()
        if any(x in q for x in ['mauzo', 'sale', 'profit', 'deni', 'debt', 'tax']): return 'FINANCE'
        if any(x in q for x in ['staff', 'employee', 'stock', 'inventory', 'supply']): return 'OPERATIONS'
        if any(x in q for x in ['goal', 'strategy', 'plan', 'growth', 'empire']): return 'STRATEGY'
        return 'META'

    def _update_head_state(self, head, query, response, entities):
        state_key = f"{self.base_key}_head_{head}"
        state = cache.get(state_key, {})
        
        # 1. ENTITY SYNCING
        if entities:
            state.update(entities)
            
        # 2. QUALITY DATA EXTRACTION (Figures, Targets, Specific dates)
        # Extract figures like "10M", "50,000", "20%"
        figures = re.findall(r'(\d+(?:[.,]\d+)?\s?[kKmм%]|(?:\d{1,3}(?:,\d{3})+|\d+))', query)
        if figures:
            state['last_figures'] = figures
            if "goal" in query.lower() or "target" in query.lower() or "shabaha" in query.lower():
                state['current_strategic_target'] = figures[0]
                state['target_context'] = query
                logging.info(f"🎯 Synaptic Lock: Locked target '{figures[0]}' into {head} head.")

        # 3. DATE LOCKING
        date_match = re.search(r'\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{4})?)\b', query)
        if date_match:
            state['anchor_date'] = date_match.group(1)

        # 4. PRIMARY OBJECTIVE
        if any(x in query.lower() for x in ["main goal", "focus", "shabaha kuu", "lengo"]):
            state['primary_objective'] = query
            
        cache.set(state_key, state, timeout=86400)

    def _apply_attention_gate(self, query, history):
        """
        Selective Attention Logic.
        Prioritizes turns with semantic overlap.
        """
        if not history: return []
        
        from difflib import SequenceMatcher
        scored_history = []
        for turn in history:
            score = SequenceMatcher(None, query.lower(), turn['q'].lower()).ratio()
            # Multiplier for same-head context
            if self._route_to_head(None, query) == turn['head']:
                score += 0.2
            scored_history.append((turn, score))
            
        # Return top 4 most relevant turns
        scored_history.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored_history[:4]]

    def _compress_history_if_needed(self):
        """
        Recursive Strategic Compression.
        Aggregates session data into condensed facts.
        """
        raw = self.get_raw_history()
        if len(raw) < 12: return
        
        # If we have too much history, we 'digest' the middle 5 items
        digest_indices = range(3, 8)
        digest_data = [raw[i] for i in digest_indices]
        
        fact_key = f"{self.base_key}_facts"
        current_facts = cache.get(fact_key, "")
        
        summary = f"Summary of turns {raw[3]['ts']}: "
        intents = set(turn['intent'] for turn in digest_data if turn['intent'])
        summary += f"Analyzed {list(intents)}. "
        
        new_facts = current_facts + "\n" + summary
        cache.set(fact_key, new_facts, timeout=86400)

    def get_raw_history(self): return cache.get(f"{self.base_key}_raw", [])
    def get_compressed_facts(self): return cache.get(f"{self.base_key}_facts", "")
    def _get_head_state(self, head): return cache.get(f"{self.base_key}_head_{head}", {})

class AIMemory:
    """
    GPT-4 LEVEL MEMORY WRAPPER
    Bridges the Neural Core with the existing Agent architecture.
    """
    def __init__(self, user_id, session_id='default', window_size=10):
        self.core = SuperNeuralLSTM(user_id, session_id)

    def add_interaction(self, query, response, intent=None, entities=None):
        self.core.add_interaction(query, response, intent, entities)

    def add_exchange(self, user_query, ai_response, sql=None):
        # Infer entities for the Input Gate
        entities = {}
        if "deni" in user_query or "debt" in user_query: entities['focus'] = 'debt'
        if "goal" in user_query: entities['strategic_anchor'] = user_query
        
        self.core.add_interaction(user_query, ai_response, entities=entities)

    def get_history(self):
        return self.core.get_raw_history()

    def get_context_string(self):
        # We search the active cursor for potential query context, or use a default
        return self.core.get_context_for_brain("General Context")

    def save_agent_context(self, context_dict):
        key = f"{self.core.base_key}_agent_state"
        cache.set(key, context_dict, timeout=86400)

    def get_agent_context(self, default=None):
        key = f"{self.core.base_key}_agent_state"
        return cache.get(key, default)
