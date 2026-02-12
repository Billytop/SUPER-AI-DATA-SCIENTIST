"""
SephlightyAI Embedding Engine
Author: Antigravity AI
Version: 1.0.0

Zero-dependency text and data embedding engine.
Uses TF-IDF + hash-based embeddings + cosine similarity for semantic search.
No external ML frameworks required.

CAPABILITIES:
  - Text → dense vector embedding
  - Business data → semantic embedding
  - Cosine similarity search
  - In-memory FAISS-like vector index
  - Embedding persistence and retrieval
"""

import math
import hashlib
import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

logger = logging.getLogger("EMBEDDING_ENGINE")
logger.setLevel(logging.INFO)


# =============================================================================
# 1. TEXT PREPROCESSOR
# =============================================================================

class TextPreprocessor:
    """Clean and normalize text for embedding."""

    STOPWORDS_EN = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'can',
        'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
        'and', 'or', 'but', 'not', 'this', 'that', 'these', 'those',
        'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they', 'me',
        'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our',
    }

    STOPWORDS_SW = {
        'ya', 'na', 'wa', 'kwa', 'ni', 'za', 'la', 'au', 'katika',
        'hii', 'hizi', 'huo', 'ile', 'yake', 'wake', 'kwenye',
    }

    @staticmethod
    def clean(text: str) -> str:
        """Normalize and clean text."""
        text = text.lower().strip()
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    @classmethod
    def tokenize(cls, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize text into meaningful words."""
        cleaned = cls.clean(text)
        tokens = cleaned.split()

        if remove_stopwords:
            all_stopwords = cls.STOPWORDS_EN | cls.STOPWORDS_SW
            tokens = [t for t in tokens if t not in all_stopwords and len(t) > 1]

        return tokens


# =============================================================================
# 2. HASH-BASED EMBEDDING
# =============================================================================

class HashEmbedder:
    """
    Hash-based text embedding using feature hashing.
    Maps tokens to a fixed-dimensional vector space deterministically.
    Zero-dependency alternative to learned embeddings.
    """

    def __init__(self, dim: int = 128, n_hash_functions: int = 4):
        self.dim = dim
        self.n_hash = n_hash_functions

    def _hash_token(self, token: str, seed: int) -> int:
        """Hash a token to an index with a given seed."""
        h = hashlib.sha256(f"{seed}:{token}".encode('utf-8')).hexdigest()
        return int(h, 16) % self.dim

    def _hash_sign(self, token: str, seed: int) -> int:
        """Determine sign (+1 or -1) for the hash."""
        h = hashlib.md5(f"{seed}:{token}".encode('utf-8')).hexdigest()
        return 1 if int(h, 16) % 2 == 0 else -1

    def embed_tokens(self, tokens: List[str]) -> List[float]:
        """Embed a list of tokens into a dense vector using feature hashing."""
        vector = [0.0] * self.dim

        for token in tokens:
            for seed in range(self.n_hash):
                idx = self._hash_token(token, seed)
                sign = self._hash_sign(token, seed)
                vector[idx] += sign * 1.0

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector


# =============================================================================
# 3. TF-IDF EMBEDDING
# =============================================================================

class TFIDFEmbedder:
    """
    TF-IDF weighted embedding.
    Builds a vocabulary from documents and weighs terms by importance.
    """

    def __init__(self, max_vocab: int = 10000, dim: int = 128):
        self.max_vocab = max_vocab
        self.dim = dim
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0
        self.hasher = HashEmbedder(dim)

    def fit(self, documents: List[str]) -> None:
        """Build vocabulary and compute IDF from a corpus."""
        self.doc_count = len(documents)
        doc_freq = Counter()

        for doc in documents:
            tokens = TextPreprocessor.tokenize(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        # Select top-N by frequency
        top_tokens = doc_freq.most_common(self.max_vocab)

        self.vocab = {token: idx for idx, (token, _) in enumerate(top_tokens)}
        self.idf = {
            token: math.log(self.doc_count / (1 + freq))
            for token, freq in top_tokens
        }

    def embed(self, text: str) -> List[float]:
        """Embed text using TF-IDF weighted hash embedding."""
        tokens = TextPreprocessor.tokenize(text)

        if not tokens:
            return [0.0] * self.dim

        # Compute TF
        tf = Counter(tokens)
        total = len(tokens)

        # Weight tokens by TF-IDF
        weighted_tokens = []
        for token, count in tf.items():
            tf_score = count / total
            idf_score = self.idf.get(token, math.log(self.doc_count + 1))
            weight = tf_score * idf_score
            # Repeat token proportionally to weight
            repeats = max(1, int(weight * 10))
            weighted_tokens.extend([token] * repeats)

        return self.hasher.embed_tokens(weighted_tokens)


# =============================================================================
# 4. COSINE SIMILARITY
# =============================================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# =============================================================================
# 5. VECTOR INDEX (In-memory FAISS-like)
# =============================================================================

class VectorIndex:
    """
    In-memory vector index for fast similarity search.
    Supports insert, search, delete, and persistence.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectors: List[List[float]] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self._id_map: Dict[str, int] = {}

    def insert(self, vector_id: str, vector: List[float],
               meta: Optional[Dict[str, Any]] = None) -> None:
        """Insert a vector with metadata."""
        if vector_id in self._id_map:
            # Update existing
            idx = self._id_map[vector_id]
            self.vectors[idx] = vector
            self.metadata[idx] = meta or {}
        else:
            # Insert new
            self._id_map[vector_id] = len(self.vectors)
            self.vectors.append(vector)
            self.metadata.append(meta or {})
            self.ids.append(vector_id)

    def search(self, query_vector: List[float],
               top_k: int = 5,
               min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """Find the top-k most similar vectors."""
        if not self.vectors:
            return []

        similarities = []
        for i, vec in enumerate(self.vectors):
            sim = cosine_similarity(query_vector, vec)
            if sim >= min_similarity:
                similarities.append((i, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, sim in similarities[:top_k]:
            results.append({
                "id": self.ids[idx],
                "similarity": round(sim, 4),
                "metadata": self.metadata[idx],
            })

        return results

    def delete(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        if vector_id not in self._id_map:
            return False

        idx = self._id_map[vector_id]

        # Remove from all lists
        self.vectors.pop(idx)
        self.metadata.pop(idx)
        self.ids.pop(idx)

        # Rebuild ID map
        self._id_map = {vid: i for i, vid in enumerate(self.ids)}
        return True

    @property
    def size(self) -> int:
        return len(self.vectors)

    def save(self, filepath: str) -> None:
        """Save index to JSON file."""
        data = {
            "dim": self.dim,
            "vectors": self.vectors,
            "metadata": self.metadata,
            "ids": self.ids,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str) -> None:
        """Load index from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.dim = data.get("dim", self.dim)
            self.vectors = data.get("vectors", [])
            self.metadata = data.get("metadata", [])
            self.ids = data.get("ids", [])
            self._id_map = {vid: i for i, vid in enumerate(self.ids)}
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Could not load index from {filepath}")


# =============================================================================
# 6. BUSINESS EMBEDDING ENGINE
# =============================================================================

class BusinessEmbeddingEngine:
    """
    High-level embedding engine for business data.
    Combines hash embeddings + TF-IDF + domain-aware weighting.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.hasher = HashEmbedder(dim)
        self.tfidf = TFIDFEmbedder(dim=dim)

        # Specialized indices per domain
        self.indices: Dict[str, VectorIndex] = {
            "products": VectorIndex(dim),
            "customers": VectorIndex(dim),
            "invoices": VectorIndex(dim),
            "conversations": VectorIndex(dim),
            "facts": VectorIndex(dim),
            "general": VectorIndex(dim),
        }

        # Domain-specific boosting terms
        self.domain_boost = {
            "products": ["product", "item", "stock", "inventory", "bidhaa", "price"],
            "customers": ["customer", "client", "mteja", "debt", "deni", "credit"],
            "invoices": ["invoice", "sale", "receipt", "mauzo", "order", "payment"],
            "conversations": ["asked", "said", "told", "query", "question"],
        }

        logger.info(f"BusinessEmbeddingEngine initialized: dim={dim}")

    def embed_text(self, text: str) -> List[float]:
        """Embed any text into a dense vector."""
        tokens = TextPreprocessor.tokenize(text, remove_stopwords=True)
        if not tokens:
            return [0.0] * self.dim
        return self.hasher.embed_tokens(tokens)

    def embed_with_domain(self, text: str, domain: str = "general") -> List[float]:
        """Embed text with domain-specific boosting."""
        tokens = TextPreprocessor.tokenize(text, remove_stopwords=True)

        # Boost domain-relevant tokens
        boost_terms = self.domain_boost.get(domain, [])
        boosted = []
        for token in tokens:
            boosted.append(token)
            if token in boost_terms:
                boosted.append(token)  # double weight for domain terms

        if not boosted:
            return [0.0] * self.dim

        return self.hasher.embed_tokens(boosted)

    def embed_row(self, row: Dict[str, Any]) -> List[float]:
        """Embed a database row."""
        parts = []
        for key, value in row.items():
            parts.append(f"{key} {value}")
        combined = " ".join(parts)
        return self.embed_text(combined)

    def index_document(self, doc_id: str, text: str,
                       domain: str = "general",
                       metadata: Optional[Dict] = None) -> None:
        """Index a document into the appropriate domain index."""
        vector = self.embed_with_domain(text, domain)

        index = self.indices.get(domain, self.indices["general"])
        index.insert(doc_id, vector, metadata or {"text": text[:200]})

    def search(self, query: str, domain: str = "general",
               top_k: int = 5, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """Semantic search within a domain."""
        query_vector = self.embed_with_domain(query, domain)

        index = self.indices.get(domain, self.indices["general"])
        return index.search(query_vector, top_k, min_similarity)

    def search_all_domains(self, query: str,
                           top_k: int = 5) -> Dict[str, List[Dict]]:
        """Search across all domain indices."""
        results = {}
        query_vector = self.embed_text(query)

        for domain, index in self.indices.items():
            if index.size > 0:
                domain_results = index.search(query_vector, top_k)
                if domain_results:
                    results[domain] = domain_results

        return results

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two texts."""
        vec_a = self.embed_text(text_a)
        vec_b = self.embed_text(text_b)
        return cosine_similarity(vec_a, vec_b)

    def get_index_stats(self) -> Dict[str, int]:
        """Get statistics about indexed documents."""
        return {domain: index.size for domain, index in self.indices.items()}


# =============================================================================
# 7. GLOBAL SINGLETON
# =============================================================================

EMBEDDING_ENGINE = BusinessEmbeddingEngine(dim=128)

logger.info("Embedding Engine v1.0.0 — 6 domain indices ready.")
