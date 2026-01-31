"""
Advanced Intent Classifier with Machine Learning
Uses pattern matching, keyword extraction, and statistical models for intent detection.
"""

from typing import Dict, List, Tuple, Optional
import re
from collections import Counter
import math


class IntentClassifier:
    """
    ML-powered intent classification with confidence scoring.
    Uses TF-IDF, cosine similarity, and pattern matching.
    """
    
    def __init__(self):
        self.training_data = self._build_training_data()
        self.intent_keywords = self._build_intent_keywords()
        self.intent_patterns = self._build_intent_patterns()
        self.idf_scores = self._calculate_idf()
        
    def classify(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Classify query intent with confidence scores.
        
        Args:
            query: User query text
            top_k: Number of top intents to return
            
        Returns:
            List of (intent, confidence) dicts sorted by confidence
        """
        query_lower = query.lower()
        
        # Get scores from multiple methods
        keyword_scores = self._keyword_scoring(query_lower)
        pattern_scores = self._pattern_matching(query_lower)
        tfidf_scores = self._tfidf_similarity(query_lower)
        
        # Combine scores with weights
        combined_scores = {}
        all_intents = set(keyword_scores.keys()) | set(pattern_scores.keys()) | set(tfidf_scores.keys())
        
        for intent in all_intents:
            keyword_score = keyword_scores.get(intent, 0.0)
            pattern_score = pattern_scores.get(intent, 0.0)
            tfidf_score = tfidf_scores.get(intent, 0.0)
            
            # Weighted combination
            combined = (
                keyword_score * 0.4 +
                pattern_score * 0.3 +
                tfidf_score * 0.3
            )
            combined_scores[intent] = combined
        
        # Sort by score
        sorted_intents = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for intent, score in sorted_intents:
            results.append({
                'intent': intent,
                'confidence': min(score, 1.0),
                'method': self._get_primary_method(intent, keyword_scores, pattern_scores, tfidf_scores)
            })
        
        return results
    
    def get_intent_features(self, query: str) -> Dict:
        """Extract features from query for intent classification."""
        query_lower = query.lower()
        tokens = self._tokenize(query_lower)
        
        return {
            'length': len(tokens),
            'has_question': '?' in query,
            'starts_with_wh': tokens[0] in ['what', 'when', 'where', 'who', 'why', 'how'] if tokens else False,
            'has_comparison': any(word in query_lower for word in ['vs', 'versus', 'compare', 'than']),
            'has_time': any(word in query_lower for word in ['today', 'yesterday', 'month', 'year', 'week']),
            'has_aggregation': any(word in query_lower for word in ['total', 'sum', 'count', 'average']),
            'has_filter': any(word in query_lower for word in ['best', 'top', 'worst', 'low', 'high']),
            'entity_types': self._detect_entity_types(query_lower)
        }
    
    def train_from_examples(self, examples: List[Tuple[str, str]]):
        """
        Train classifier from new examples.
        
        Args:
            examples: List of (query, intent) tuples
        """
        for query, intent in examples:
            if intent not in self.training_data:
                self.training_data[intent] = []
            self.training_data[intent].append(query.lower())
        
        # Recalculate IDF scores
        self.idf_scores = self._calculate_idf()
    
    def explain_classification(self, query: str) -> Dict:
        """
        Explain why a particular intent was chosen.
        
        Args:
            query: User query
            
        Returns:
            Dict with explanation details
        """
        results = self.classify(query, top_k=1)
        if not results:
            return {'explanation': 'No intent matched'}
        
        top_intent = results[0]
        query_lower = query.lower()
        
        # Find matching keywords
        matched_keywords = []
        if top_intent['intent'] in self.intent_keywords:
            for keyword in self.intent_keywords[top_intent['intent']]:
                if keyword in query_lower:
                    matched_keywords.append(keyword)
        
        # Find matching patterns
        matched_patterns = []
        if top_intent['intent'] in self.intent_patterns:
            for pattern_info in self.intent_patterns[top_intent['intent']]:
                if re.search(pattern_info['pattern'], query_lower):
                    matched_patterns.append(pattern_info['description'])
        
        return {
            'intent': top_intent['intent'],
            'confidence': top_intent['confidence'],
            'matched_keywords': matched_keywords,
            'matched_patterns': matched_patterns,
            'explanation': self._generate_explanation(
                top_intent['intent'],
                matched_keywords,
                matched_patterns,
                top_intent['confidence']
            )
        }
    
    def _keyword_scoring(self, query: str) -> Dict[str, float]:
        """Score intents based on keyword matching."""
        scores = {}
        tokens = set(self._tokenize(query))
        
        for intent, keywords in self.intent_keywords.items():
            matches = sum(1 for kw in keywords if kw in query)
            if matches > 0:
                # Normalize by total keywords for this intent
                scores[intent] = matches / len(keywords)
        
        return scores
    
    def _pattern_matching(self, query: str) -> Dict[str, float]:
        """Score intents based on regex pattern matching."""
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            max_score = 0.0
            for pattern_info in patterns:
                if re.search(pattern_info['pattern'], query):
                    max_score = max(max_score, pattern_info['weight'])
            
            if max_score > 0:
                scores[intent] = max_score
        
        return scores
    
    def _tfidf_similarity(self, query: str) -> Dict[str, float]:
        """Calculate TF-IDF based similarity with training examples."""
        query_tokens = self._tokenize(query)
        query_tf = self._calculate_tf(query_tokens)
        query_vector = self._create_tfidf_vector(query_tf)
        
        scores = {}
        for intent, examples in self.training_data.items():
            max_similarity = 0.0
            
            for example in examples:
                example_tokens = self._tokenize(example)
                example_tf = self._calculate_tf(example_tokens)
                example_vector = self._create_tfidf_vector(example_tf)
                
                similarity = self._cosine_similarity(query_vector, example_vector)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity > 0:
                scores[intent] = max_similarity
        
        return scores
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return [word for word in text.split() if len(word) > 1]
    
    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate term frequency."""
        if not tokens:
            return {}
        
        counter = Counter(tokens)
        total = len(tokens)
        
        return {term: count / total for term, count in counter.items()}
    
    def _calculate_idf(self) -> Dict[str, float]:
        """Calculate inverse document frequency across all training data."""
        all_docs = []
        for examples in self.training_data.values():
            all_docs.extend(examples)
        
        if not all_docs:
            return {}
        
        # Count documents containing each term
        term_doc_count = Counter()
        for doc in all_docs:
            unique_terms = set(self._tokenize(doc))
            term_doc_count.update(unique_terms)
        
        # Calculate IDF
        num_docs = len(all_docs)
        idf = {}
        for term, doc_count in term_doc_count.items():
            idf[term] = math.log(num_docs / (1 + doc_count))
        
        return idf
    
    def _create_tfidf_vector(self, tf: Dict[str, float]) -> Dict[str, float]:
        """Create TF-IDF vector."""
        tfidf = {}
        for term, tf_score in tf.items():
            idf_score = self.idf_scores.get(term, 0.0)
            tfidf[term] = tf_score * idf_score
        
        return tfidf
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        # Get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _detect_entity_types(self, query: str) -> List[str]:
        """Detect entity types mentioned in query."""
        entities = []
        
        if any(word in query for word in ['product', 'item', 'goods', 'bidhaa']):
            entities.append('product')
        if any(word in query for word in ['customer', 'client', 'mteja']):
            entities.append('customer')
        if any(word in query for word in ['employee', 'staff', 'mfanyakazi']):
            entities.append('employee')
        if any(word in query for word in ['today', 'yesterday', 'month', 'year', 'leo', 'jana']):
            entities.append('date')
        
        return entities
    
    def _get_primary_method(self, intent: str, keyword_scores: Dict, pattern_scores: Dict, tfidf_scores: Dict) -> str:
        """Determine which method contributed most to classification."""
        kw = keyword_scores.get(intent, 0)
        pat = pattern_scores.get(intent, 0)
        tf = tfidf_scores.get(intent, 0)
        
        if kw >= pat and kw >= tf:
            return 'keyword'
        elif pat >= tf:
            return 'pattern'
        else:
            return 'similarity'
    
    def _generate_explanation(self, intent: str, keywords: List, patterns: List, confidence: float) -> str:
        """Generate human-readable explanation."""
        parts = [f"Classified as '{intent}' with {confidence:.0%} confidence."]
        
        if keywords:
            parts.append(f"Matched keywords: {', '.join(keywords[:3])}")
        if patterns:
            parts.append(f"Matched patterns: {', '.join(patterns[:2])}")
        
        return " ".join(parts)
    
    def _build_training_data(self) -> Dict[str, List[str]]:
        """Build training dataset with example queries per intent."""
        return {
            'SALES': [
                'sales today', 'total sales', 'revenue this month', 'mauzo leo',
                'how much did I sell', 'sales performance', 'income report',
                'show me sales', 'what are my sales', 'sales figures'
            ],
            'INVENTORY': [
                'stock levels', 'inventory value', 'products in stock', 'mzigo',
                'what items do i have', 'check inventory', 'stock count',
                'available products', 'warehouse stock'
            ],
            'CUSTOMER_RISK': [
                'customer debt', 'who owes me', 'outstanding payments', 'madeni',
                'customers with debt', 'unpaid invoices', 'payment due',
                'debt aging', 'credit customers'
            ],
            'BEST_PRODUCT': [
                'best selling product', 'top products', 'most sold items',
                'best performers', 'highest selling', 'popular products'
            ],
            'EMPLOYEE': [
                'employee performance', 'staff sales', 'best employee',
                'worker productivity', 'team performance', 'mfanyakazi bora'
            ],
            'COMPARISON': [
                'compare sales', 'this month vs last month', 'linganisha',
                'sales comparison', 'year over year', 'period comparison'
            ],
            'FORECAST': [
                'forecast sales', 'predict revenue', 'future sales',
                'sales projection', 'next month estimate'
            ],
            'EXPENSE': [
                'expenses', 'costs', 'spending', 'matumizi', 'gharama',
                'how much did i spend', 'expense report'
            ]
        }
    
    def _build_intent_keywords(self) -> Dict[str, List[str]]:
        """Build keyword lists per intent."""
        return {
            'SALES': ['sales', 'revenue', 'income', 'mauzo', 'sell', 'sold', 'turnover'],
            'INVENTORY': ['stock', 'inventory', 'goods', 'mzigo', 'warehouse', 'products', 'items'],
            'CUSTOMER_RISK': ['debt', 'owe', 'outstanding', 'madeni', 'unpaid', 'due', 'credit'],
            'BEST_PRODUCT': ['best', 'top', 'most', 'popular', 'highest', 'leading'],
            'EMPLOYEE': ['employee', 'staff', 'worker', 'mfanyakazi', 'team', 'personnel'],
            'COMPARISON': ['compare', 'vs', 'versus', 'linganisha', 'than', 'against'],
            'FORECAST': ['forecast', 'predict', 'future', 'projection', 'estimate', 'utabiri'],
            'EXPENSE': ['expense', 'cost', 'spending', 'matumizi', 'gharama', 'spent']
        }
    
    def _build_intent_patterns(self) -> Dict[str, List[Dict]]:
        """Build regex patterns per intent."""
        return {
            'SALES': [
                {'pattern': r'\b(sales?|revenue|mauzo)\s+(today|this|last)', 'weight': 0.9, 'description': 'Sales with time period'},
                {'pattern': r'\btotal\s+sales?\b', 'weight': 0.85, 'description': 'Total sales query'},
            ],
            'INVENTORY': [
                {'pattern': r'\b(stock|inventory|mzigo)\s+(level|value|count)', 'weight': 0.9, 'description': 'Inventory metrics'},
                {'pattern': r'\blow\s+stock\b', 'weight': 0.95, 'description': 'Low stock query'},
            ],
            'CUSTOMER_RISK': [
                {'pattern': r'\b(customer|client|mteja)s?\s+(debt|owe|madeni)', 'weight': 0.9, 'description': 'Customer debt query'},
                {'pattern': r'\bdebt\s+aging\b', 'weight': 0.95, 'description': 'Debt aging'},
            ],
            'COMPARISON': [
                {'pattern': r'\bcompare\s+\w+\s+(and|vs|versus)\b', 'weight': 0.95, 'description': 'Direct comparison'},
                {'pattern': r'\bthis\s+\w+\s+vs\s+last\b', 'weight': 0.9, 'description': 'Time period comparison'},
            ],
            'BEST_PRODUCT': [
                {'pattern': r'\b(best|top|most)\s+(selling|sold|popular)\b', 'weight': 0.95, 'description': 'Best product query'},
            ]
        }
