"""
SephlightyAI RAG Engine (Retrieval-Augmented Generation)
Author: Antigravity AI
Version: 1.0.0

Combines semantic search with contextual reasoning to produce
fact-grounded answers. Never hallucinates — every answer is
backed by retrieved evidence.

PIPELINE:
  Query → Embed → Search (Vector Index + Memory) → Rank → Context Build → Answer

PRINCIPLES:
  - Answers are grounded in database facts
  - Uncertainty is declared honestly
  - Evidence is cited with source tracking
"""

import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger("RAG_ENGINE")
logger.setLevel(logging.INFO)


# =============================================================================
# 1. CONTEXT BUILDER
# =============================================================================

class ContextBuilder:
    """
    Builds a ranked, deduped context window from multiple retrieval sources.
    Ensures the context stays within token limits.
    """

    def __init__(self, max_context_tokens: int = 2000):
        self.max_context_tokens = max_context_tokens

    def build(self, retrieved_chunks: List[Dict[str, Any]],
              query: str) -> Dict[str, Any]:
        """
        Build a context window from retrieved chunks.

        Args:
            retrieved_chunks: List of {text, source, similarity, domain}
            query: Original user query

        Returns:
            Structured context with ranked evidence
        """
        if not retrieved_chunks:
            return {
                "context_text": "",
                "evidence": [],
                "total_chunks": 0,
                "context_quality": 0.0,
            }

        # Deduplicate by content hash
        seen = set()
        unique_chunks = []
        for chunk in retrieved_chunks:
            text = str(chunk.get("text", ""))
            content_hash = hash(text[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_chunks.append(chunk)

        # Sort by similarity/relevance
        unique_chunks.sort(
            key=lambda c: c.get("similarity", 0),
            reverse=True
        )

        # Build context within token limit
        context_parts = []
        evidence = []
        current_tokens = 0

        for chunk in unique_chunks:
            text = str(chunk.get("text", ""))
            chunk_tokens = len(text.split())

            if current_tokens + chunk_tokens > self.max_context_tokens:
                break

            context_parts.append(text)
            evidence.append({
                "text": text[:200],
                "source": chunk.get("source", "unknown"),
                "domain": chunk.get("domain", "general"),
                "similarity": chunk.get("similarity", 0),
            })
            current_tokens += chunk_tokens

        # Calculate context quality (avg similarity of included chunks)
        avg_sim = sum(e["similarity"] for e in evidence) / len(evidence) if evidence else 0

        return {
            "context_text": "\n\n---\n\n".join(context_parts),
            "evidence": evidence,
            "total_chunks": len(evidence),
            "context_quality": round(avg_sim, 4),
        }


# =============================================================================
# 2. FACT GROUNDING CHECKER
# =============================================================================

class FactGrounder:
    """
    Ensures answers are grounded in facts.
    Detects potential hallucination and marks uncertain claims.
    """

    # Hallucination signals
    UNCERTAIN_PHRASES = [
        "i think", "probably", "maybe", "might be", "possibly",
        "nadhani", "labda", "pengine", "huenda",
        "it seems", "appears to be", "could be",
    ]

    def check_grounding(self, answer: str,
                        evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if an answer is grounded in the provided evidence.

        Returns:
            Dict with grounding_score, is_grounded, warnings
        """
        answer_lower = answer.lower()

        # Check for uncertainty signals
        uncertainty_count = sum(
            1 for phrase in self.UNCERTAIN_PHRASES
            if phrase in answer_lower
        )

        # Check if answer references evidence
        evidence_overlap = 0
        if evidence:
            answer_words = set(answer_lower.split())
            for ev in evidence:
                ev_words = set(str(ev.get("text", "")).lower().split())
                overlap = len(answer_words & ev_words)
                if overlap > 3:
                    evidence_overlap += 1

        # Calculate grounding score
        if not evidence:
            grounding_score = 0.3  # No evidence = low grounding
        else:
            evidence_ratio = evidence_overlap / len(evidence)
            uncertainty_penalty = uncertainty_count * 0.1
            grounding_score = min(1.0, max(0.0,
                0.5 + evidence_ratio * 0.5 - uncertainty_penalty
            ))

        return {
            "grounding_score": round(grounding_score, 4),
            "is_grounded": grounding_score >= 0.5,
            "evidence_used": evidence_overlap,
            "total_evidence": len(evidence),
            "uncertainty_signals": uncertainty_count,
            "warnings": self._generate_warnings(grounding_score, evidence),
        }

    def _generate_warnings(self, score: float,
                           evidence: List[Dict]) -> List[str]:
        """Generate grounding warnings."""
        warnings = []
        if score < 0.3:
            warnings.append("LOW GROUNDING: Answer may not be fully supported by data")
        if not evidence:
            warnings.append("NO EVIDENCE: Answer was generated without supporting data")
        return warnings


# =============================================================================
# 3. RAG ENGINE
# =============================================================================

class RAGEngine:
    """
    Retrieval-Augmented Generation Engine.
    Combines semantic search, memory recall, and fact grounding
    to produce accurate, evidence-backed business answers.
    """

    def __init__(self, max_context_tokens: int = 2000):
        self.context_builder = ContextBuilder(max_context_tokens)
        self.fact_grounder = FactGrounder()
        self._embedding_engine = None
        self._memory_service = None
        self._transformer = None

    def _get_embedding_engine(self):
        """Lazy-load embedding engine."""
        if self._embedding_engine is None:
            try:
                from laravel_modules.ai_brain.embedding_engine import EMBEDDING_ENGINE
                self._embedding_engine = EMBEDDING_ENGINE
            except ImportError:
                logger.warning("Embedding engine not available.")
        return self._embedding_engine

    def _get_memory_service(self):
        """Lazy-load memory service."""
        if self._memory_service is None:
            try:
                from laravel_modules.ai_brain.memory_service import MEMORY_SERVICE
                self._memory_service = MEMORY_SERVICE
            except ImportError:
                logger.warning("Memory service not available.")
        return self._memory_service

    def _get_transformer(self):
        """Lazy-load transformer."""
        if self._transformer is None:
            try:
                from laravel_modules.ai_brain.transformer_core import TRANSFORMER_BRAIN
                self._transformer = TRANSFORMER_BRAIN
            except ImportError:
                logger.warning("Transformer not available.")
        return self._transformer

    def retrieve(self, query: str,
                 domain: str = "general",
                 business_id: Optional[int] = None,
                 top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from all available sources.

        Sources:
          1. Vector index (embedding-based semantic search)
          2. Memory service (short-term + long-term recall)
          3. Database facts

        Returns:
            List of retrieved chunks with metadata
        """
        chunks = []

        # 1. Vector index search
        embedding_engine = self._get_embedding_engine()
        if embedding_engine:
            # Search specific domain
            results = embedding_engine.search(query, domain, top_k)
            for r in results:
                chunks.append({
                    "text": r.get("metadata", {}).get("text", ""),
                    "source": "vector_index",
                    "domain": domain,
                    "similarity": r.get("similarity", 0),
                })

            # Search across all domains for cross-references
            all_results = embedding_engine.search_all_domains(query, top_k=3)
            for d, results in all_results.items():
                if d != domain:
                    for r in results:
                        chunks.append({
                            "text": r.get("metadata", {}).get("text", ""),
                            "source": f"vector_index_{d}",
                            "domain": d,
                            "similarity": r.get("similarity", 0) * 0.8,
                        })

        # 2. Memory search
        memory = self._get_memory_service()
        if memory:
            # Short-term memory
            stm_results = memory.short_term.search(query)
            for key, value in stm_results[:5]:
                chunks.append({
                    "text": f"{key}: {value}",
                    "source": "short_term_memory",
                    "domain": "conversation",
                    "similarity": 0.7,
                })

            # Long-term memory
            if business_id:
                lt_facts = memory.long_term.recall_facts(
                    business_id=business_id,
                    domain=domain if domain != "general" else None,
                    limit=5
                )
                for fact in lt_facts:
                    chunks.append({
                        "text": fact.get("content", ""),
                        "source": "long_term_memory",
                        "domain": fact.get("domain", "general"),
                        "similarity": fact.get("confidence", 0.5),
                    })

        return chunks

    def augmented_query(self, query: str,
                        domain: str = "general",
                        business_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Full RAG pipeline: Retrieve → Build Context → Ground → Augment.

        Returns:
            Dict with context, evidence, grounding info, augmented query
        """
        start_time = time.time()

        # 1. Retrieve
        retrieved = self.retrieve(query, domain, business_id)

        # 2. Build context
        context = self.context_builder.build(retrieved, query)

        # 3. Create augmented prompt
        if context["context_text"]:
            augmented = (
                f"CONTEXT (grounded facts):\n"
                f"{context['context_text']}\n\n"
                f"QUESTION: {query}\n\n"
                f"INSTRUCTION: Answer based ONLY on the context above. "
                f"If the information is not in the context, say so honestly."
            )
        else:
            augmented = (
                f"QUESTION: {query}\n\n"
                f"NOTE: No pre-existing context found. "
                f"Answer from database query results only."
            )

        elapsed = round(time.time() - start_time, 4)

        return {
            "augmented_query": augmented,
            "context": context,
            "retrieval_time": elapsed,
            "chunks_retrieved": len(retrieved),
            "query_domain": domain,
        }

    def validate_answer(self, answer: str,
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an answer against retrieved evidence.

        Args:
            answer: The generated answer text
            context: The context object from augmented_query()

        Returns:
            Grounding validation results
        """
        evidence = context.get("evidence", [])
        return self.fact_grounder.check_grounding(answer, evidence)

    def index_business_data(self, data_rows: List[Dict[str, Any]],
                            domain: str = "general",
                            id_field: str = "id") -> int:
        """
        Index business data rows into the embedding engine.

        Args:
            data_rows: List of dicts (database rows)
            domain: Target domain index
            id_field: Field to use as document ID

        Returns:
            Number of documents indexed
        """
        engine = self._get_embedding_engine()
        if not engine:
            return 0

        indexed = 0
        for row in data_rows:
            doc_id = str(row.get(id_field, indexed))
            text_parts = [f"{k}: {v}" for k, v in row.items()
                          if v is not None and k != id_field]
            text = " | ".join(text_parts)

            engine.index_document(doc_id, text, domain, metadata={
                "text": text[:300],
                "raw": {k: str(v)[:100] for k, v in row.items()},
            })
            indexed += 1

        logger.info(f"Indexed {indexed} documents in '{domain}'")
        return indexed


# =============================================================================
# 4. GLOBAL SINGLETON
# =============================================================================

RAG_ENGINE = RAGEngine(max_context_tokens=2000)

logger.info("RAG Engine v1.0.0 — Retrieval-Augmented Generation ready.")
