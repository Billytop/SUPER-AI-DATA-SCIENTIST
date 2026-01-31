"""
Semantic Similarity Matcher
Intelligent query matching using synonym recognition and semantic understanding.
"""

from difflib import SequenceMatcher
from typing import List, Dict, Tuple
import re


class SemanticMatcher:
    """
    Matches user queries to known patterns using semantic similarity.
    Handles synonyms, related terms, and conceptual matching.
    """
    
    def __init__(self):
        self.synonym_map = self._build_synonym_map()
        self.intent_patterns = self._build_intent_patterns()
        
    def find_similar_queries(self, query: str, known_queries: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find similar queries from a list of known queries.
        
        Args:
            query: User's input query
            known_queries: List of known/training queries
            threshold: Minimum similarity score
            
        Returns:
            List of (query, score) tuples sorted by similarity
        """
        # Normalize queries
        query_normalized = self._normalize_query(query)
        
        similarities = []
        for known_query in known_queries:
            known_normalized = self._normalize_query(known_query)
            score = self._semantic_similarity(query_normalized, known_normalized)
            
            if score >= threshold:
                similarities.append((known_query, score))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        """
        Generate query variants using synonyms.
        
        Args:
            query: Original query
            
        Returns:
            List of query variants
        """
        variants = [query]
        words = query.lower().split()
        
        # Replace each word with its synonyms
        for i, word in enumerate(words):
            if word in self.synonym_map:
                for synonym in self.synonym_map[word]:
                    variant_words = words.copy()
                    variant_words[i] = synonym
                    variants.append(' '.join(variant_words))
        
        return list(set(variants))[:10]  # Return unique variants, max 10
    
    def match_intent_by_pattern(self, query: str) -> Dict[str, float]:
        """
        Match query to intents using pattern matching.
        
        Args:
            query: User query
            
        Returns:
            Dict mapping intent to confidence score
        """
        query_lower = query.lower()
        query_normalized = self._normalize_query(query)
        
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            max_score = 0.0
            
            for pattern in patterns:
                # Keyword match
                if isinstance(pattern, str):
                    if pattern in query_lower:
                        max_score = max(max_score, 0.9)
                    else:
                        # Fuzzy match
                        similarity = SequenceMatcher(None, pattern, query_normalized).ratio()
                        max_score = max(max_score, similarity * 0.7)
                
                # Regex pattern match
                elif isinstance(pattern, tuple) and pattern[0] == 'regex':
                    if re.search(pattern[1], query_lower):
                        max_score = max(max_score, 0.95)
            
            if max_score > 0:
                intent_scores[intent] = max_score
        
        return intent_scores
    
    def cluster_similar_queries(self, queries: List[str], threshold: float = 0.7) -> List[List[str]]:
        """
        Group similar queries into clusters.
        
        Args:
            queries: List of queries to cluster
            threshold: Similarity threshold for clustering
            
        Returns:
            List of query clusters
        """
        clusters = []
        processed = set()
        
        for i, query1 in enumerate(queries):
            if i in processed:
                continue
            
            cluster = [query1]
            processed.add(i)
            
            for j, query2 in enumerate(queries[i+1:], start=i+1):
                if j in processed:
                    continue
                
                similarity = self._semantic_similarity(
                    self._normalize_query(query1),
                    self._normalize_query(query2)
                )
                
                if similarity >= threshold:
                    cluster.append(query2)
                    processed.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison."""
        # Lowercase
        q = query.lower()
        
        # Remove punctuation
        q = re.sub(r'[^\w\s]', ' ', q)
        
        # Replace synonyms with canonical form
        words = q.split()
        normalized_words = []
        
        for word in words:
            # Find canonical form
            canonical = word
            for key, synonyms in self.synonym_map.items():
                if word in synonyms or word == key:
                    canonical = key
                    break
            normalized_words.append(canonical)
        
        return ' '.join(normalized_words)
    
    def _semantic_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate semantic similarity between two queries.
        
        Uses combination of:
        - Exact word overlap
        - Sequence similarity
        - Synonym matching
        """
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        # Word overlap score
        if not words1 or not words2:
            overlap_score = 0.0
        else:
            overlap = len(words1 & words2)
            overlap_score = (2.0 * overlap) / (len(words1) + len(words2))
        
        # Sequence similarity score
        sequence_score = SequenceMatcher(None, query1, query2).ratio()
        
        # Combined score (weighted average)
        combined_score = (overlap_score * 0.6) + (sequence_score * 0.4)
        
        return combined_score
    
    def _build_synonym_map(self) -> Dict[str, List[str]]:
        """Build comprehensive synonym mapping."""
        return {
            # Sales
            'sales': ['revenue', 'income', 'earnings', 'turnover', 'mauzo'],
            'sell': ['sold', 'sale', 'selling'],
            
            # Purchases
            'purchase': ['buy', 'bought', 'procurement', 'manunuzi'],
            'buying': ['purchasing', 'procurement'],
            
            # Expenses
            'expense': ['cost', 'expenditure', 'spending', 'matumizi', 'gharama'],
            'spent': ['expended', 'consumed'],
            
            # Products
            'product': ['item', 'goods', 'merchandise', 'bidhaa'],
            'stock': ['inventory', 'goods', 'mzigo'],
            
            # Customers
            'customer': ['client', 'buyer', 'mteja', 'wateja'],
            
            # Employees
            'employee': ['staff', 'worker', 'personnel', 'mfanyakazi'],
            
            # Financial
            'profit': ['margin', 'earnings', 'gain', 'faida'],
            'loss': ['deficit', 'shortfall', 'hasara'],
            'debt': ['payable', 'outstanding', 'owed', 'deni', 'madeni'],
            
            # Time
            'today': ['now', 'current', 'leo'],
            'yesterday': ['jana', 'previous day'],
            'month': ['mwezi', 'monthly'],
            'year': ['mwaka', 'annual', 'yearly'],
            
            # Actions
            'show': ['display', 'give', 'present', 'onyesha'],
            'list': ['enumerate', 'show all', 'orodha'],
            'compare': ['versus', 'vs', 'linganisha'],
            'analyze': ['examine', 'review', 'study'],
            
            # Quantifiers
            'best': ['top', 'highest', 'maximum', 'bora'],
            'worst': ['lowest', 'minimum', 'bottom'],
            'most': ['maximum', 'highest'],
            'least': ['minimum', 'lowest'],
            
            # Metrics
            'total': ['sum', 'aggregate', 'jumla'],
            'average': ['mean', 'typical', 'wastani'],
            'count': ['number', 'quantity', 'idadi'],
        }
    
    def _build_intent_patterns(self) -> Dict[str, List]:
        """Build intent matching patterns."""
        return {
            'SALES': [
                'sales', 'revenue', 'income', 'mauzo',
                ('regex', r'\btotal\s+sales\b'),
                ('regex', r'\bsales\s+(today|this|last)\b'),
            ],
            'PURCHASES': [
                'purchase', 'buy', 'procurement', 'manunuzi',
                ('regex', r'\btotal\s+purchase'),
            ],
            'INVENTORY': [
                'stock', 'inventory', 'goods', 'mzigo',
                ('regex', r'\bstock\s+(level|value|count)\b'),
            ],
            'EMPLOYEE': [
                'employee', 'staff', 'worker', 'mfanyakazi',
                ('regex', r'\bbest\s+(employee|staff)\b'),
            ],
            'CUSTOMER_RISK': [
                'debt', 'outstanding', 'payable', 'deni', 'madeni',
                ('regex', r'\b(debt|outstanding)\s+(customer|client)\b'),
            ],
            'COMPARISON': [
                'compare', 'versus', 'vs', 'linganisha',
                ('regex', r'\bcompare\s+\w+\s+(and|vs|versus)\b'),
            ],
            'FORECAST': [
                'forecast', 'predict', 'projection', 'utabiri',
                ('regex', r'\b(next|future|upcoming)\s+(month|year)\b'),
            ],
            'BEST_PRODUCT': [
                ('regex', r'\bbest\s+product'),
                ('regex', r'\btop\s+(selling|product)'),
            ],
        }
