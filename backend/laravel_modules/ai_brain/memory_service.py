"""
SephlightyAI Memory Service
Author: Antigravity AI
Version: 1.0.0

Manages all AI memory operations: short-term, long-term, and summary memory.
Provides memory recall for the Agent Pipeline and RAG system.

MEMORY TIERS:
  1. Short-Term Memory (session) — in-memory, lost on restart
  2. Long-Term Memory (DB) — persistent facts and conversation summaries
  3. Summary Memory — weekly/monthly aggregated insights
"""

import logging
import time
import hashlib
import math
import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger("MEMORY_SERVICE")
logger.setLevel(logging.INFO)


# =============================================================================
# 1. SHORT-TERM MEMORY (Session-scoped, in-memory)
# =============================================================================

class ShortTermMemory:
    """
    In-memory conversation context for the current session.
    LRU-evicted when it exceeds max capacity.
    """

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._store: OrderedDict = OrderedDict()
        self._context_stack: List[Dict[str, Any]] = []

    def remember(self, key: str, value: Any) -> None:
        """Store a short-term memory entry."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = {
            "value": value,
            "timestamp": time.time(),
        }
        # Evict oldest if over capacity
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def recall(self, key: str) -> Optional[Any]:
        """Recall a short-term memory by key."""
        entry = self._store.get(key)
        if entry:
            self._store.move_to_end(key)
            return entry["value"]
        return None

    def push_context(self, context: Dict[str, Any]) -> None:
        """Push conversation context onto the stack."""
        self._context_stack.append({
            **context,
            "_timestamp": time.time(),
        })
        # Keep only last 20 contexts
        if len(self._context_stack) > 20:
            self._context_stack = self._context_stack[-20:]

    def get_recent_context(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the N most recent conversation contexts."""
        return self._context_stack[-n:]

    def search(self, query: str) -> List[Tuple[str, Any]]:
        """Simple keyword search over short-term memories."""
        query_lower = query.lower()
        results = []
        for key, entry in self._store.items():
            value_str = str(entry["value"]).lower()
            if query_lower in key.lower() or query_lower in value_str:
                results.append((key, entry["value"]))
        return results

    def clear(self) -> None:
        """Clear all short-term memory."""
        self._store.clear()
        self._context_stack.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# =============================================================================
# 2. LONG-TERM MEMORY SERVICE (DB-backed)
# =============================================================================

class LongTermMemory:
    """
    Persistent memory backed by Django models.
    Handles conversation summaries, business facts, and decision logs.
    """

    def __init__(self):
        self._models_loaded = False
        self._ConversationMemory = None
        self._MemoryChunk = None
        self._DecisionLog = None
        self._CustomerProfile = None
        self._ProductProfile = None

    def _load_models(self):
        """Lazy-load Django models to avoid import issues at module level."""
        if self._models_loaded:
            return True
        try:
            from intelligence.ai_memory_models import (
                AIConversationMemory,
                AIMemoryChunk,
                AIDecisionLog,
                AICustomerProfile,
                AIProductProfile,
            )
            self._ConversationMemory = AIConversationMemory
            self._MemoryChunk = AIMemoryChunk
            self._DecisionLog = AIDecisionLog
            self._CustomerProfile = AICustomerProfile
            self._ProductProfile = AIProductProfile
            self._models_loaded = True
            return True
        except Exception as e:
            logger.warning(f"Could not load AI memory models: {e}")
            return False

    # --- Conversation Memory ---

    def store_conversation_summary(self, business_id: int, summary: str,
                                    topics: List[str], entities: List[str],
                                    insights: List[str],
                                    conversation_id: Optional[str] = None,
                                    user_id: Optional[int] = None,
                                    importance: float = 0.5) -> Optional[str]:
        """Store a conversation summary in long-term memory."""
        if not self._load_models():
            return None

        try:
            from core.models import Business
            business = Business.objects.get(id=business_id)
            user = None
            if user_id:
                from django.contrib.auth.models import User
                user = User.objects.filter(id=user_id).first()

            import uuid as uuid_mod
            conv_uuid = uuid_mod.UUID(conversation_id) if conversation_id else None

            memory = self._ConversationMemory.objects.create(
                business=business,
                user=user,
                conversation_id=conv_uuid,
                summary=summary,
                key_topics=topics,
                key_entities=entities,
                key_insights=insights,
                importance_score=importance,
            )
            logger.info(f"Stored conversation memory: {memory.id}")
            return str(memory.id)
        except Exception as e:
            logger.error(f"Failed to store conversation memory: {e}")
            return None

    def recall_conversations(self, business_id: int,
                              topic: Optional[str] = None,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """Recall past conversation summaries."""
        if not self._load_models():
            return []

        try:
            qs = self._ConversationMemory.objects.filter(
                business_id=business_id
            ).order_by('-importance_score', '-created_at')

            if topic:
                qs = qs.filter(key_topics__contains=topic)

            return list(qs.values(
                'id', 'summary', 'key_topics', 'key_entities',
                'key_insights', 'importance_score', 'created_at'
            )[:limit])
        except Exception as e:
            logger.error(f"Failed to recall conversations: {e}")
            return []

    # --- Memory Chunks (Business Facts) ---

    def store_fact(self, business_id: int, content: str,
                   structured_data: Dict = None,
                   domain: str = "general",
                   chunk_type: str = "fact",
                   source: str = "computed",
                   source_query: Optional[str] = None,
                   confidence: float = 1.0,
                   fact_date: Optional[str] = None) -> Optional[str]:
        """Store an immutable business fact."""
        if not self._load_models():
            return None

        try:
            from core.models import Business
            business = Business.objects.get(id=business_id)

            chunk = self._MemoryChunk.objects.create(
                business=business,
                chunk_type=chunk_type,
                content=content,
                structured_data=structured_data or {},
                domain=domain,
                source=source,
                source_query=source_query,
                confidence=confidence,
                fact_date=fact_date,
            )
            logger.info(f"Stored fact: [{chunk_type}] {content[:50]}")
            return str(chunk.id)
        except Exception as e:
            logger.error(f"Failed to store fact: {e}")
            return None

    def recall_facts(self, business_id: int,
                     domain: Optional[str] = None,
                     chunk_type: Optional[str] = None,
                     limit: int = 20,
                     active_only: bool = True) -> List[Dict[str, Any]]:
        """Recall business facts from long-term memory."""
        if not self._load_models():
            return []

        try:
            qs = self._MemoryChunk.objects.filter(business_id=business_id)

            if active_only:
                qs = qs.filter(is_active=True)
            if domain:
                qs = qs.filter(domain=domain)
            if chunk_type:
                qs = qs.filter(chunk_type=chunk_type)

            return list(qs.values(
                'id', 'chunk_type', 'content', 'structured_data',
                'domain', 'confidence', 'created_at', 'fact_date'
            ).order_by('-created_at')[:limit])
        except Exception as e:
            logger.error(f"Failed to recall facts: {e}")
            return []

    def store_user_correction(self, business_id: int, original_fact_id: str,
                               correction: str, corrected_data: Dict = None) -> Optional[str]:
        """Store a user correction (supersedes original fact)."""
        if not self._load_models():
            return None

        try:
            import uuid as uuid_mod
            original = self._MemoryChunk.objects.get(id=uuid_mod.UUID(original_fact_id))

            # Create correction chunk
            new_chunk = self._MemoryChunk.objects.create(
                business_id=business_id,
                chunk_type='correction',
                content=correction,
                structured_data=corrected_data or {},
                domain=original.domain,
                source='user_stated',
                confidence=1.0,
            )

            # Mark original as superseded
            original.is_active = False
            original.superseded_by = new_chunk
            original.save()

            return str(new_chunk.id)
        except Exception as e:
            logger.error(f"Failed to store correction: {e}")
            return None

    # --- Decision Log ---

    def log_decision(self, business_id: int, domain: str,
                     decision: str, reasoning: str,
                     input_query: Optional[str] = None,
                     input_data: Dict = None,
                     confidence: float = 0.8,
                     user_id: Optional[int] = None) -> Optional[str]:
        """Log an AI decision with reasoning."""
        if not self._load_models():
            return None

        try:
            from core.models import Business
            business = Business.objects.get(id=business_id)
            user = None
            if user_id:
                from django.contrib.auth.models import User
                user = User.objects.filter(id=user_id).first()

            decision_obj = self._DecisionLog.objects.create(
                business=business,
                user=user,
                domain=domain,
                decision=decision,
                reasoning=reasoning,
                input_query=input_query,
                input_data=input_data or {},
                confidence=confidence,
            )
            logger.info(f"Logged decision: {decision[:50]}")
            return str(decision_obj.id)
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
            return None

    def get_decision_history(self, business_id: int,
                              domain: Optional[str] = None,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """Get past AI decisions."""
        if not self._load_models():
            return []

        try:
            qs = self._DecisionLog.objects.filter(business_id=business_id)
            if domain:
                qs = qs.filter(domain=domain)

            return list(qs.values(
                'id', 'domain', 'decision', 'reasoning',
                'confidence', 'was_accepted', 'created_at', 'version'
            ).order_by('-created_at')[:limit])
        except Exception as e:
            logger.error(f"Failed to get decision history: {e}")
            return []

    # --- Customer Profiles ---

    def update_customer_profile(self, business_id: int, contact_id: int,
                                 **profile_data) -> Optional[str]:
        """Create or update a customer intelligence profile."""
        if not self._load_models():
            return None

        try:
            from core.models import Business
            business = Business.objects.get(id=business_id)

            profile, created = self._CustomerProfile.objects.update_or_create(
                business=business,
                contact_id=contact_id,
                defaults=profile_data,
            )
            action = "Created" if created else "Updated"
            logger.info(f"{action} customer profile for contact {contact_id}")
            return str(profile.id)
        except Exception as e:
            logger.error(f"Failed to update customer profile: {e}")
            return None

    def get_customer_profile(self, business_id: int,
                              contact_id: int) -> Optional[Dict[str, Any]]:
        """Get a customer's AI profile."""
        if not self._load_models():
            return None

        try:
            profile = self._CustomerProfile.objects.filter(
                business_id=business_id, contact_id=contact_id
            ).values().first()
            return profile
        except Exception as e:
            logger.error(f"Failed to get customer profile: {e}")
            return None

    def get_high_risk_customers(self, business_id: int,
                                 min_risk: float = 0.5,
                                 limit: int = 20) -> List[Dict[str, Any]]:
        """Get customers with high risk scores."""
        if not self._load_models():
            return []

        try:
            return list(self._CustomerProfile.objects.filter(
                business_id=business_id,
                risk_score__gte=min_risk
            ).values().order_by('-risk_score')[:limit])
        except Exception as e:
            logger.error(f"Failed to get high-risk customers: {e}")
            return []

    # --- Product Profiles ---

    def update_product_profile(self, business_id: int, product_id: int,
                                **profile_data) -> Optional[str]:
        """Create or update a product intelligence profile."""
        if not self._load_models():
            return None

        try:
            from core.models import Business
            business = Business.objects.get(id=business_id)

            profile, created = self._ProductProfile.objects.update_or_create(
                business=business,
                product_id=product_id,
                defaults=profile_data,
            )
            action = "Created" if created else "Updated"
            logger.info(f"{action} product profile for product {product_id}")
            return str(profile.id)
        except Exception as e:
            logger.error(f"Failed to update product profile: {e}")
            return None

    def get_dead_stock(self, business_id: int,
                       limit: int = 50) -> List[Dict[str, Any]]:
        """Get products flagged as dead stock."""
        if not self._load_models():
            return []

        try:
            return list(self._ProductProfile.objects.filter(
                business_id=business_id,
                is_dead_stock=True
            ).values().order_by('days_since_last_sale')[:limit])
        except Exception as e:
            logger.error(f"Failed to get dead stock: {e}")
            return []


# =============================================================================
# 3. SUMMARY MEMORY (Aggregated insights)
# =============================================================================

class SummaryMemory:
    """
    Generates and stores periodic summary insights.
    Weekly/monthly digests of AI learnings and patterns.
    """

    def __init__(self, long_term: LongTermMemory):
        self.long_term = long_term

    def generate_weekly_summary(self, business_id: int) -> Optional[str]:
        """Generate a weekly summary from recent facts and decisions."""
        facts = self.long_term.recall_facts(
            business_id=business_id,
            limit=50,
        )
        decisions = self.long_term.get_decision_history(
            business_id=business_id,
            limit=20,
        )

        if not facts and not decisions:
            return None

        # Count facts by domain
        domain_counts = {}
        for fact in facts:
            domain = fact.get('domain', 'general')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Summarize
        summary_parts = [f"Weekly AI Summary:"]
        summary_parts.append(f"  - {len(facts)} new facts stored")
        summary_parts.append(f"  - {len(decisions)} decisions made")

        for domain, count in domain_counts.items():
            summary_parts.append(f"  - {domain}: {count} facts")

        summary = "\n".join(summary_parts)

        # Store as a high-importance memory
        self.long_term.store_fact(
            business_id=business_id,
            content=summary,
            structured_data={
                "type": "weekly_summary",
                "facts_count": len(facts),
                "decisions_count": len(decisions),
                "domain_breakdown": domain_counts,
            },
            domain="summary",
            chunk_type="pattern",
            source="summarizer",
            importance=0.8,
        )

        return summary


# =============================================================================
# 4. UNIFIED MEMORY SERVICE
# =============================================================================

class MemoryService:
    """
    Unified memory interface for the AI brain.
    Combines short-term, long-term, and summary memory.
    """

    def __init__(self):
        self.short_term = ShortTermMemory(max_entries=200)
        self.long_term = LongTermMemory()
        self.summary = SummaryMemory(self.long_term)
        logger.info("MemoryService initialized: 3 memory tiers active.")

    def remember(self, key: str, value: Any) -> None:
        """Quick short-term memory store."""
        self.short_term.remember(key, value)

    def recall(self, key: str) -> Optional[Any]:
        """Quick short-term memory recall."""
        return self.short_term.recall(key)

    def store_context(self, query: str, answer: str,
                      domain: str = "general",
                      metadata: Dict = None) -> None:
        """Store conversation context for continuity."""
        self.short_term.push_context({
            "query": query,
            "answer": answer,
            "domain": domain,
            "metadata": metadata or {},
        })

    def get_context_window(self, n: int = 5) -> List[Dict]:
        """Get recent conversation context."""
        return self.short_term.get_recent_context(n)

    def search_memory(self, query: str,
                      business_id: Optional[int] = None) -> Dict[str, Any]:
        """Search across all memory tiers."""
        results = {
            "short_term": self.short_term.search(query),
            "long_term_facts": [],
            "long_term_conversations": [],
        }

        if business_id:
            results["long_term_facts"] = self.long_term.recall_facts(
                business_id=business_id, limit=5
            )
            results["long_term_conversations"] = self.long_term.recall_conversations(
                business_id=business_id, limit=5
            )

        return results


# =============================================================================
# 5. GLOBAL SINGLETON
# =============================================================================

MEMORY_SERVICE = MemoryService()

logger.info("Memory Service v1.0.0 — 3 tiers active (short-term, long-term, summary).")
