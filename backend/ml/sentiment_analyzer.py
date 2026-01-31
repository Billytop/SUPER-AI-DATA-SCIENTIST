"""
Sentiment Analysis Engine
Analyzes emotional tone and sentiment of user queries and feedback.
"""

from typing import Dict, List
import re


class SentimentAnalyzer:
    """
    Analyzes sentiment and emotional tone of text.
    """
    
    def __init__(self):
        self.positive_words = self._build_positive_lexicon()
        self.negative_words = self._build_negative_lexicon()
        self.intensifiers = self._build_intensifiers()
        self.negations = ['not', 'no', 'never', 'neither', 'nobody', 'nothing', 'hapana', 'siyo']
        
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment score, label, and details
        """
        text_lower = text.lower()
        tokens = self._tokenize(text_lower)
        
        # Calculate sentiment scores
        positive_score = 0.0
        negative_score = 0.0
        
        for i, token in enumerate(tokens):
            # Check for negation
            is_negated = i > 0 and tokens[i-1] in self.negations
            
            # Check intensifiers
            intensifier = 1.0
            if i > 0 and tokens[i-1] in self.intensifiers:
                intensifier = 1.5
            
            # Calculate scores
            if token in self.positive_words:
                score = self.positive_words[token] * intensifier
                if is_negated:
                    negative_score += score
                else:
                    positive_score += score
            
            elif token in self.negative_words:
                score = self.negative_words[token] * intensifier
                if is_negated:
                    positive_score += score
                else:
                    negative_score += score
        
        # Normalize scores
        total_score = positive_score + negative_score
        if total_score > 0:
            positive_score = positive_score / total_score
            negative_score = negative_score / total_score
        
        # Calculate compound score (-1 to 1)
        compound = positive_score - negative_score
        
        # Determine sentiment label
        if compound >= 0.3:
            label = 'positive'
            emoji = '😊'
        elif compound <= -0.3:
            label = 'negative'
            emoji = '😞'
        else:
            label = 'neutral'
            emoji = '😐'
        
        return {
            'compound': compound,
            'positive': positive_score,
            'negative': negative_score,
            'neutral': 1 - abs(compound),
            'label': label,
            'emoji': emoji,
            'intensity': self._get_intensity(abs(compound))
        }
    
    def detect_urgency(self, text: str) -> Dict:
        """Detect urgency level in text."""
        text_lower = text.lower()
        
        urgent_keywords = [
            'urgent', 'asap', 'immediately', 'emergency', 'critical',
            'haraka', 'dharura', 'sasa hivi', 'now', 'quick'
        ]
        
        urgency_score = sum(1 for word in urgent_keywords if word in text_lower)
        
        # Check for exclamation marks
        if '!' in text:
            urgency_score += text.count('!')
        
        # Check for all caps words
        words = text.split()
        caps_count = sum(1 for word in words if word.isupper() and len(word) > 2)
        urgency_score += caps_count
        
        if urgency_score >= 3:
            level = 'high'
        elif urgency_score >= 1:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': min(urgency_score / 5.0, 1.0),
            'indicators': urgency_score
        }
    
    def detect_satisfaction(self, text: str) -> Dict:
        """Detect customer satisfaction level."""
        sentiment = self.analyze(text)
        
        satisfaction_keywords = {
            'high': ['excellent', 'great', 'perfect', 'amazing', 'wonderful', 'fantastic', 'nzuri sana'],
            'medium': ['good', 'nice', 'okay', 'fine', 'nzuri', 'vizuri'],
            'low': ['bad', 'poor', 'terrible', 'awful', 'mbaya', 'vibaya']
        }
        
        text_lower = text.lower()
        
        for level, keywords in satisfaction_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        'level': level,
                        'sentiment': sentiment['label'],
                        'confidence': 0.8
                    }
        
        # Infer from sentiment
        if sentiment['compound'] >= 0.5:
            level = 'high'
        elif sentiment['compound'] >= 0:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'sentiment': sentiment['label'],
            'confidence': 0.6
        }
    
    def analyze_feedback(self, feedback: str) -> Dict:
        """Comprehensive feedback analysis."""
        sentiment = self.analyze(feedback)
        urgency = self.detect_urgency(feedback)
        satisfaction = self.detect_satisfaction(feedback)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(feedback)
        
        # Determine action needed
        if sentiment['label'] == 'negative' and urgency['level'] == 'high':
            action = 'immediate_response_required'
            priority = 'urgent'
        elif sentiment['label'] == 'negative':
            action = 'follow_up_needed'
            priority = 'high'
        elif urgency['level'] == 'high':
            action = 'quick_response_needed'
            priority = 'medium'
        else:
            action = 'acknowledge'
            priority = 'normal'
        
        return {
            'sentiment': sentiment,
            'urgency': urgency,
            'satisfaction': satisfaction,
            'key_phrases': key_phrases,
            'recommended_action': action,
            'priority': priority
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        text = re.sub(r'[^\w\s]', ' ', text)
        return [word for word in text.split() if len(word) > 1]
    
    def _get_intensity(self, score: float) -> str:
        """Get intensity label."""
        if score >= 0.7:
            return 'strong'
        elif score >= 0.4:
            return 'moderate'
        else:
            return 'weak'
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from feedback."""
        # Simplified phrase extraction
        phrases = []
        
        # Common problem indicators
        problem_patterns = [
            r'(not working|broken|error|issue|problem|fail)',
            r'(slow|takes long|waiting|delay)',
            r'(missing|lost|cannot find)',
        ]
        
        for pattern in problem_patterns:
            matches = re.findall(pattern, text.lower())
            phrases.extend(matches)
        
        return list(set(phrases))[:5]
    
    def _build_positive_lexicon(self) -> Dict[str, float]:
        """Build positive word lexicon with scores."""
        return {
            # English
            'excellent': 1.0, 'amazing': 1.0, 'wonderful': 1.0, 'fantastic': 1.0,
            'great': 0.9, 'good': 0.7, 'nice': 0.6, 'fine': 0.5, 'okay': 0.4,
            'happy': 0.8, 'love': 0.9, 'like': 0.6, 'enjoy': 0.7,
            'perfect': 1.0, 'best': 0.9, 'better': 0.7, 'improve': 0.6,
            'thank': 0.7, 'thanks': 0.7, 'appreciate': 0.8,
            # Swahili
            'nzuri': 0.7, 'vizuri': 0.7, 'sana': 0.5, 'poa': 0.7,
            'asante': 0.7, 'nakupenda': 0.8, 'safi': 0.8,
        }
    
    def _build_negative_lexicon(self) -> Dict[str, float]:
        """Build negative word lexicon with scores."""
        return {
            # English
            'terrible': 1.0, 'awful': 1.0, 'horrible': 1.0, 'worst': 1.0,
            'bad': 0.8, 'poor': 0.7, 'wrong': 0.6, 'fail': 0.8, 'failed': 0.8,
            'error': 0.6, 'problem': 0.7, 'issue': 0.5, 'broken': 0.8,
            'slow': 0.5, 'delay': 0.6, 'waiting': 0.4, 'difficult': 0.6,
            'hate': 0.9, 'dislike': 0.7, 'annoying': 0.7, 'frustrated': 0.8,
            # Swahili
            'mbaya': 0.8, 'vibaya': 0.8, 'tatizo': 0.7, 'shida': 0.7,
            'kosa': 0.6, 'haifanyi': 0.7, 'haiwezi': 0.6,
        }
    
    def _build_intensifiers(self) -> List[str]:
        """Build intensifier word list."""
        return [
            'very', 'really', 'extremely', 'so', 'too', 'quite',
            'absolutely', 'completely', 'totally', 'highly',
            'sana', 'kabisa'
        ]
