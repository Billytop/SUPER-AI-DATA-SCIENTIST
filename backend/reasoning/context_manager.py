"""
Multi-Turn Conversation Context Manager
Tracks conversation history and resolves references across multiple queries.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json


class ContextManager:
    """
    Manages conversation context across multiple turns.
    Handles pronoun resolution, previous query references, and context retention.
    """
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or self._generate_session_id()
        self.conversation_history = []
        self.context_data = {}
        self.last_entities = {}
        self.last_intent = None
        self.last_response = None
        self.session_start = datetime.now()
        
    def add_query(self, query: str, intent: str, entities: Dict[str, Any], response: str):
        """
        Add a query-response pair to conversation history.
        
        Args:
            query: User's query text
            intent: Detected intent
            entities: Extracted entities
            response: AI's response
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'intent': intent,
            'entities': entities,
            'response': response[:500]  # Store truncated response
        }
        
        self.conversation_history.append(entry)
        self.last_entities = entities
        self.last_intent = intent
        self.last_response = response
        
        # Keep only last 10 turns
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def resolve_references(self, query: str) -> str:
        """
        Resolve pronouns and references to previous queries.
        
        Args:
            query: Current query with potential references
            
        Returns:
            Query with references resolved
        """
        if not self.conversation_history:
            return query
        
        resolved_query = query
        query_lower = query.lower()
        
        # Pronoun resolution patterns
        pronoun_patterns = {
            'it': self._resolve_it,
            'that': self._resolve_that,
            'them': self._resolve_them,
            'those': self._resolve_those,
            'this': self._resolve_this,
            'these': self._resolve_these,
        }
        
        for pronoun, resolver in pronoun_patterns.items():
            if pronoun in query_lower.split():
                resolved = resolver()
                if resolved:
                    # Replace the pronoun with the resolved entity
                    resolved_query = resolved_query.replace(pronoun, resolved)
                    resolved_query = resolved_query.replace(pronoun.title(), resolved)
        
        # Reference phrases
        reference_phrases = {
            'same product': self._get_last_product,
            'same customer': self._get_last_customer,
            'same period': self._get_last_date_range,
            'previous query': self._get_last_query,
            'last time': self._get_last_date_range,
        }
        
        for phrase, resolver in reference_phrases.items():
            if phrase in query_lower:
                resolved = resolver()
                if resolved:
                    resolved_query = resolved_query.replace(phrase, resolved)
        
        return resolved_query
    
    def get_context_hints(self) -> Dict[str, Any]:
        """
        Get contextual hints for better understanding current query.
        
        Returns:
            Dict with context information like recent topics, entities, patterns
        """
        if not self.conversation_history:
            return {}
        
        # Analyze recent conversation
        recent_intents = [h['intent'] for h in self.conversation_history[-3:]]
        recent_products = []
        recent_customers = []
        
        for h in self.conversation_history[-3:]:
            if h['entities'].get('products'):
                recent_products.extend([p.get('name') for p in h['entities']['products']])
            if h['entities'].get('customers'):
                recent_customers.extend([c.get('name') for c in h['entities']['customers']])
        
        return {
            'recent_intents': recent_intents,
            'recent_products': list(set(recent_products)),
            'recent_customers': list(set(recent_customers)),
            'conversation_length': len(self.conversation_history),
            'session_duration': (datetime.now() - self.session_start).seconds,
            'last_intent': self.last_intent
        }
    
    def suggest_follow_ups(self) -> List[str]:
        """
        Generate contextually relevant follow-up suggestions.
        
        Returns:
            List of suggested follow-up queries
        """
        if not self.last_intent:
            return []
        
        suggestions = []
        
        # Intent-based suggestions
        intent_suggestions = {
            'SALES': [
                "📊 Break down by category?",
                "📈 Show sales trend?",
                "📅 Compare with last month?"
            ],
            'BEST_PRODUCT': [
                "💰 Show by revenue instead?",
                "📅 Seasonal patterns?",
                "🔍 Customer demographics?"
            ],
            'CUSTOMER_RISK': [
                "📧 Send payment reminders?",
                "📊 Show aging analysis?",
                "📈 Collection rate trend?"
            ],
            'INVENTORY': [
                "📉 Show low stock items?",
                "💰 Value by category?",
                "📊 Turnover rate?"
            ],
            'EMPLOYEE': [
                "📊 Performance comparison?",
                "💰 Commission breakdown?",
                "📈 Sales trend?"
            ],
            'COMPARISON': [
                "📅 Add more periods?",
                "📊 Break down by category?",
                "📈 Show percentage change?"
            ]
        }
        
        suggestions = intent_suggestions.get(self.last_intent, [
            "💡 What else can I help with?",
            "📊 Want to see more details?",
            "📈 Need historical comparison?"
        ])
        
        # Entity-based suggestions
        if self.last_entities.get('products'):
            product = self.last_entities['products'][0].get('name')
            suggestions.insert(0, f"🔍 Show details for {product}?")
        
        if self.last_entities.get('customers'):
            customer = self.last_entities['customers'][0].get('name')
            suggestions.insert(0, f"👤 Show {customer}'s history?")
        
        return suggestions[:3]  # Return top 3
    
    def is_refinement_query(self, query: str) -> bool:
        """
        Detect if current query is refining/modifying previous query.
        
        Args:
            query: Current query
            
        Returns:
            True if this is a refinement
        """
        refinement_indicators = [
            'by', 'instead', 'change', 'show', 'with', 'without', 'add', 'remove',
            'more', 'less', 'only', 'except', 'just', 'filter', 'sort', 'group'
        ]
        
        query_lower = query.lower()
        words = query_lower.split()
        
        # Short queries starting with refinement words
        if len(words) <= 4 and any(words[0] == word for word in refinement_indicators):
            return True
        
        # Contains reference to previous query
        reference_phrases = ['same', 'that', 'it', 'those', 'these', 'previous', 'last']
        if any(phrase in query_lower for phrase in reference_phrases):
            return True
        
        return False
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of entire conversation session.
        
        Returns:
            Dict with session statistics and highlights
        """
        if not self.conversation_history:
            return {'status': 'empty'}
        
        intents = [h['intent'] for h in self.conversation_history]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            'session_id': self.session_id,
            'duration_seconds': (datetime.now() - self.session_start).seconds,
            'query_count': len(self.conversation_history),
            'intents_used': intent_counts,
            'most_common_intent': max(intent_counts, key=intent_counts.get) if intent_counts else None,
            'first_query': self.conversation_history[0]['query'],
            'last_query': self.conversation_history[-1]['query']
        }
    
    def clear_context(self):
        """Clear conversation history and reset context."""
        self.conversation_history = []
        self.context_data = {}
        self.last_entities = {}
        self.last_intent = None
        self.last_response = None
        self.session_start = datetime.now()
    
    # Private helper methods for pronoun resolution
    
    def _resolve_it(self) -> Optional[str]:
        """Resolve 'it' to last mentioned product or entity."""
        if self.last_entities.get('products'):
            return self.last_entities['products'][0].get('name')
        if self.last_entities.get('customers'):
            return self.last_entities['customers'][0].get('name')
        return None
    
    def _resolve_that(self) -> Optional[str]:
        """Resolve 'that' similar to 'it'."""
        return self._resolve_it()
    
    def _resolve_them(self) -> Optional[str]:
        """Resolve 'them' to last mentioned products (plural)."""
        if self.last_entities.get('products') and len(self.last_entities['products']) > 1:
            products = [p.get('name') for p in self.last_entities['products'][:3]]
            return ' and '.join(products)
        return None
    
    def _resolve_those(self) -> Optional[str]:
        """Resolve 'those' similar to 'them'."""
        return self._resolve_them()
    
    def _resolve_this(self) -> Optional[str]:
        """Resolve 'this' to last mentioned single entity."""
        return self._resolve_it()
    
    def _resolve_these(self) -> Optional[str]:
        """Resolve 'these' to last mentioned entities (plural)."""
        return self._resolve_them()
    
    def _get_last_product(self) -> Optional[str]:
        """Get last mentioned product name."""
        if self.last_entities.get('products'):
            return self.last_entities['products'][0].get('name')
        return None
    
    def _get_last_customer(self) -> Optional[str]:
        """Get last mentioned customer name."""
        if self.last_entities.get('customers'):
            return self.last_entities['customers'][0].get('name')
        return None
    
    def _get_last_date_range(self) -> Optional[str]:
        """Get last mentioned date range."""
        if self.last_entities.get('dates'):
            date_entity = self.last_entities['dates'][0]
            return date_entity.get('label', date_entity.get('value'))
        return None
    
    def _get_last_query(self) -> Optional[str]:
        """Get the previous query text."""
        if len(self.conversation_history) >= 2:
            return self.conversation_history[-2]['query']
        return None
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
