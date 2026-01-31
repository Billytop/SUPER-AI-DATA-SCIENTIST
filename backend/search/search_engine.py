"""
Full-Text Search Engine
Advanced search with indexing, ranking, filters, and autocomplete.
"""

from typing import Dict, List, Optional, Set
from datetime import datetime
import re
from collections import defaultdict
import math


class SearchIndex:
    """
    Full-text search index with TF-IDF ranking.
    """
    
    def __init__(self):
        self.documents = {}
        self.inverted_index = defaultdict(set)
        self.doc_frequencies = defaultdict(int)
        self.total_docs = 0
        self.doc_lengths = {}
        
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """Add document to search index."""
        # Tokenize content
        tokens = self._tokenize(content)
        
        # Store document
        self.documents[doc_id] = {
            'content': content,
            'tokens': tokens,
            'metadata': metadata or {},
            'indexed_at': datetime.now().isoformat()
        }
        
        # Update inverted index
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.inverted_index[token].add(doc_id)
            self.doc_frequencies[token] += 1
        
        self.doc_lengths[doc_id] = len(tokens)
        self.total_docs += 1
    
    def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[Dict]:
        """
        Search documents using TF-IDF ranking.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            filters: Metadata filters
            
        Returns:
            List of search results with scores
        """
        query_tokens = self._tokenize(query)
        
        # Get candidate documents (documents containing any query term)
        candidates = set()
        for token in query_tokens:
            candidates.update(self.inverted_index.get(token, set()))
        
        # Apply filters
        if filters:
            candidates = self._apply_filters(candidates, filters)
        
        if not candidates:
            return []
        
        # Calculate TF-IDF scores
        scores = {}
        for doc_id in candidates:
            scores[doc_id] = self._calculate_tfidf_score(query_tokens, doc_id)
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for doc_id, score in ranked:
            doc = self.documents[doc_id]
            results.append({
                'doc_id': doc_id,
                'score': score,
                'content': doc['content'],
                'snippet': self._generate_snippet(doc['content'], query_tokens),
                'metadata': doc['metadata']
            })
        
        return results
    
    def autocomplete(self, prefix: str, limit: int = 10) -> List[str]:
        """Get autocomplete suggestions."""
        prefix_lower = prefix.lower()
        
        suggestions = []
        for token in self.inverted_index.keys():
            if token.startswith(prefix_lower):
                suggestions.append(token)
        
        # Sort by frequency
        suggestions.sort(key=lambda x: self.doc_frequencies[x], reverse=True)
        
        return suggestions[:limit]
    
    def suggest_similar(self, query: str, limit: int = 5) -> List[str]:
        """Suggest similar queries based on indexed content."""
        # Simple implementation - find common phrases
        query_tokens = self._tokenize(query)
        
        similar_docs = self.search(query, top_k=10)
        
        # Extract common phrases from similar documents
        phrases = set()
        for doc in similar_docs:
            content_tokens = self._tokenize(doc['content'])
            # Extract 2-word phrases
            for i in range(len(content_tokens) - 1):
                phrase = f"{content_tokens[i]} {content_tokens[i+1]}"
                phrases.add(phrase)
        
        return list(phrases)[:limit]
    
    def delete_document(self, doc_id: str) -> bool:
        """Remove document from index."""
        if doc_id not in self.documents:
            return False
        
        # Get tokens
        tokens = self.documents[doc_id]['tokens']
        
        # Update inverted index
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if doc_id in self.inverted_index[token]:
                self.inverted_index[token].remove(doc_id)
                self.doc_frequencies[token] -= 1
                
                # Remove empty entries
                if not self.inverted_index[token]:
                    del self.inverted_index[token]
                    del self.doc_frequencies[token]
        
        # Remove document
        del self.documents[doc_id]
        del self.doc_lengths[doc_id]
        self.total_docs -= 1
        
        return True
    
    def get_index_stats(self) -> Dict:
        """Get search index statistics."""
        return {
            'total_documents': self.total_docs,
            'total_terms': len(self.inverted_index),
            'avg_doc_length': sum(self.doc_lengths.values()) / self.total_docs if self.total_docs > 0 else 0,
            'largest_doc': max(self.doc_lengths.values()) if self.doc_lengths else 0,
            'smallest_doc': min(self.doc_lengths.values()) if self.doc_lengths else 0
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into searchable terms."""
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split and filter short words
        tokens = [word for word in text.split() if len(word) > 1]
        
        return tokens
    
    def _calculate_tfidf_score(self, query_tokens: List[str], doc_id: str) -> float:
        """Calculate TF-IDF score for document."""
        doc_tokens = self.documents[doc_id]['tokens']
        doc_length = self.doc_lengths[doc_id]
        
        score = 0.0
        for term in query_tokens:
            # Term frequency in document
            tf = doc_tokens.count(term) / doc_length if doc_length > 0 else 0
            
            # Inverse document frequency
            df = self.doc_frequencies.get(term, 0)
            idf = math.log((self.total_docs + 1) / (df + 1)) if df > 0 else 0
            
            score += tf * idf
        
        return score
    
    def _apply_filters(self, candidates: Set[str], filters: Dict) -> Set[str]:
        """Apply metadata filters to candidate documents."""
        filtered = set()
        
        for doc_id in candidates:
            doc = self.documents[doc_id]
            metadata = doc['metadata']
            
            # Check all filters
            matches = True
            for key, value in filters.items():
                if key not in metadata or metadata[key] != value:
                    matches = False
                    break
            
            if matches:
                filtered.add(doc_id)
        
        return filtered
    
    def _generate_snippet(self, content: str, query_tokens: List[str], max_length: int = 150) -> str:
        """Generate search result snippet with highlighted query terms."""
        # Find first occurrence of any query term
        content_lower = content.lower()
        
        min_pos = len(content)
        for token in query_tokens:
            pos = content_lower.find(token)
            if pos != -1 and pos < min_pos:
                min_pos = pos
        
        if min_pos == len(content):
            # No query terms found, return beginning
            return content[:max_length] + ('...' if len(content) > max_length else '')
        
        # Extract snippet around first match
        start = max(0, min_pos - 50)
        end = min(len(content), min_pos + 100)
        
        snippet = content[start:end]
        
        if start > 0:
            snippet = '...' + snippet
        if end < len(content):
            snippet = snippet + '...'
        
        return snippet


class AdvancedFilterBuilder:
    """
    Builds complex search filters with AND/OR/NOT logic.
    """
    
    def __init__(self):
        self.filters = []
        
    def add_filter(self, field: str, operator: str, value: any):
        """Add a filter condition."""
        self.filters.append({
            'field': field,
            'operator': operator,
            'value': value
        })
        return self
    
    def equals(self, field: str, value: any):
        """Add equals filter."""
        return self.add_filter(field, '==', value)
    
    def not_equals(self, field: str, value: any):
        """Add not equals filter."""
        return self.add_filter(field, '!=', value)
    
    def greater_than(self, field: str, value: float):
        """Add greater than filter."""
        return self.add_filter(field, '>', value)
    
    def less_than(self, field: str, value: float):
        """Add less than filter."""
        return self.add_filter(field, '<', value)
    
    def contains(self, field: str, value: str):
        """Add contains filter."""
        return self.add_filter(field, 'contains', value)
    
    def in_list(self, field: str, values: List):
        """Add in list filter."""
        return self.add_filter(field, 'in', values)
    
    def between(self, field: str, min_val: float, max_val: float):
        """Add between filter."""
        return self.add_filter(field, 'between', [min_val, max_val])
    
    def evaluate(self, document: Dict) -> bool:
        """Evaluate if document matches all filters."""
        for filter_spec in self.filters:
            if not self._evaluate_filter(filter_spec, document):
                return False
        return True
    
    def _evaluate_filter(self, filter_spec: Dict, document: Dict) -> bool:
        """Evaluate single filter."""
        field = filter_spec['field']
        operator = filter_spec['operator']
        value = filter_spec['value']
        
        if field not in document:
            return False
        
        doc_value = document[field]
        
        if operator == '==':
            return doc_value == value
        elif operator == '!=':
            return doc_value != value
        elif operator == '>':
            return doc_value > value
        elif operator == '<':
            return doc_value < value
        elif operator == 'contains':
            return value in str(doc_value)
        elif operator == 'in':
            return doc_value in value
        elif operator == 'between':
            return value[0] <= doc_value <= value[1]
        else:
            return False
    
    def build(self) -> List[Dict]:
        """Get built filters."""
        return self.filters


class FacetedSearch:
    """
    Faceted search with aggregations and drill-down.
    """
    
    def __init__(self, search_index: SearchIndex):
        self.search_index = search_index
        
    def search_with_facets(self, query: str, facet_fields: List[str]) -> Dict:
        """Search with facet aggregations."""
        # Perform base search
        results = self.search_index.search(query, top_k=1000)
        
        # Calculate facets
        facets = {}
        for field in facet_fields:
            facets[field] = self._calculate_facet(results, field)
        
        return {
            'results': results[:10],  # Top 10 results
            'total_results': len(results),
            'facets': facets
        }
    
    def _calculate_facet(self, results: List[Dict], field: str) -> Dict:
        """Calculate facet counts for a field."""
        counts = defaultdict(int)
        
        for result in results:
            metadata = result.get('metadata', {})
            if field in metadata:
                value = metadata[field]
                counts[value] += 1
        
        # Sort by count
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'field': field,
            'values': [{'value': v, 'count': c} for v, c in sorted_counts[:20]]
        }


class SearchHistory:
    """
    Tracks search history and provides insights.
    """
    
    def __init__(self):
        self.history = []
        
    def record_search(self, query: str, results_count: int, user_id: str = None):
        """Record a search query."""
        self.history.append({
            'query': query,
            'results_count': results_count,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict]:
        """Get most popular search queries."""
        query_counts = defaultdict(int)
        
        for search in self.history:
            query_counts[search['query']] += 1
        
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{'query': q, 'count': c} for q, c in sorted_queries[:limit]]
    
    def get_user_search_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get search history for a user."""
        user_searches = [s for s in self.history if s.get('user_id') == user_id]
        return user_searches[-limit:]
    
    def get_zero_result_queries(self) -> List[str]:
        """Get queries that returned no results."""
        zero_results = [s['query'] for s in self.history if s['results_count'] == 0]
        return list(set(zero_results))
