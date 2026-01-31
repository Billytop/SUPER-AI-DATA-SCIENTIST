"""
Language Detection and Translation
Detects language of queries and provides translation capabilities.
"""

from typing import Dict, List, Tuple, Optional
import re


class LanguageDetector:
    """
    Detects language and provides basic translation.
    """
    
    def __init__(self):
        self.language_patterns = self._build_language_patterns()
        self.translations = self._build_translation_dict()
        self.supported_languages = ['en', 'sw']  # English, Swahili
        
    def detect(self, text: str) -> Dict:
        """
        Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with language code, name, and confidence
        """
        text_lower = text.lower()
        tokens = text_lower.split()
        
        # Count language-specific markers
        language_scores = {'en': 0.0, 'sw': 0.0}
        
        for lang, patterns in self.language_patterns.items():
            for pattern in patterns['words']:
                count = sum(1 for token in tokens if pattern in token or token == pattern)
                language_scores[lang] += count * 2  # Word match worth more
            
            for pattern in patterns['prefixes']:
                count = sum(1 for token in tokens if token.startswith(pattern))
                language_scores[lang] += count
            
            for pattern in patterns['suffixes']:
                count = sum(1 for token in tokens if token.endswith(pattern))
                language_scores[lang] += count
        
        # Normalize scores
        total = sum(language_scores.values())
        if total > 0:
            for lang in language_scores:
                language_scores[lang] /= total
        
        # Get top language
        detected_lang = max(language_scores, key=language_scores.get)
        confidence = language_scores[detected_lang]
        
        # If confidence is low, assume English as default
        if confidence < 0.3:
            detected_lang = 'en'
            confidence = 0.5
        
        return {
            'code': detected_lang,
            'name': 'English' if detected_lang == 'en' else 'Swahili',
            'confidence': confidence,
            'scores': language_scores
        }
    
    def translate(self, text: str, from_lang: str = 'auto', to_lang: str = 'en') -> Dict:
        """
        Translate text between English and Swahili.
        
        Args:
            text: Text to translate
            from_lang: Source language ('auto', 'en', 'sw')
            to_lang: Target language ('en', 'sw')
            
        Returns:
            Dict with translated text and metadata
        """
        # Auto-detect source language if needed
        if from_lang == 'auto':
            detection = self.detect(text)
            from_lang = detection['code']
        
        # No translation needed if same language
        if from_lang == to_lang:
            return {
                'original': text,
                'translated': text,
                'from_lang': from_lang,
                'to_lang': to_lang,
                'method': 'none'
            }
        
        # Perform translation
        translated = self._word_by_word_translate(text, from_lang, to_lang)
        
        return {
            'original': text,
            'translated': translated,
            'from_lang': from_lang,
            'to_lang': to_lang,
            'method': 'dictionary'
        }
    
    def is_multilingual(self, text: str) -> bool:
        """Check if text contains multiple languages."""
        detection = self.detect(text)
        
        # If no clear winner (low confidence), it's likely mixed
        return detection['confidence'] < 0.6
    
    def get_dominant_language(self, queries: List[str]) -> str:
        """Get dominant language across multiple queries."""
        language_counts = {'en': 0, 'sw': 0}
        
        for query in queries:
            detection = self.detect(query)
            language_counts[detection['code']] += 1
        
        return max(language_counts, key=language_counts.get)
    
    def _word_by_word_translate(self, text: str, from_lang: str, to_lang: str) -> str:
        """Simple word-by-word translation."""
        words = text.split()
        translated_words = []
        
        translation_key = f"{from_lang}_{to_lang}"
        translation_dict = self.translations.get(translation_key, {})
        
        for word in words:
            word_lower = word.lower()
            
            # Check if word has translation
            if word_lower in translation_dict:
                translated_words.append(translation_dict[word_lower])
            else:
                # Keep original if no translation
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def _build_language_patterns(self) -> Dict:
        """Build language detection patterns."""
        return {
            'en': {
                'words': [
                    'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
                    'what', 'when', 'where', 'who', 'why', 'how',
                    'sales', 'customer', 'product', 'total', 'show'
                ],
                'prefixes': ['un', 're', 'pre'],
                'suffixes': ['ing', 'ed', 'tion', 'ness', 'ful']
            },
            'sw': {
                'words': [
                    'ni', 'wa', 'ya', 'la', 'cha', 'za', 'pa', 'ku',
                    'gani', 'nini', 'wapi', 'lini', 'kwa', 'nini',
                    'mauzo', 'mteja', 'bidhaa', 'jumla', 'onyesha',
                    'leo', 'jana', 'kesho', 'wiki', 'mwezi', 'mwaka'
                ],
                'prefixes': ['ki', 'vi', 'u', 'm', 'wa'],
                'suffixes': ['wa', 'ni', 'ka', 'tu', 'li']
            }
        }
    
    def _build_translation_dict(self) -> Dict:
        """Build bidirectional translation dictionary."""
        en_to_sw = {
            # Time
            'today': 'leo', 'yesterday': 'jana', 'tomorrow': 'kesho',
            'week': 'wiki', 'month': 'mwezi', 'year': 'mwaka',
            'day': 'siku', 'time': 'wakati',
            # Business
            'sales': 'mauzo', 'customer': 'mteja', 'product': 'bidhaa',
            'money': 'pesa', 'price': 'bei', 'total': 'jumla',
            'profit': 'faida', 'loss': 'hasara', 'debt': 'deni',
            'stock': 'mzigo', 'employee': 'mfanyakazi',
            # Actions
            'show': 'onyesha', 'give': 'pa', 'compare': 'linganisha',
            'buy': 'nunua', 'sell': 'uza', 'pay': 'lipa',
            # Questions
            'what': 'nini', 'when': 'lini', 'where': 'wapi',
            'who': 'nani', 'how': 'vipi', 'why': 'kwa nini',
            # Numbers (1-10)
            'one': 'moja', 'two': 'mbili', 'three':  'tatu',
            'four': 'nne', 'five': 'tano', 'six': 'sita',
            'seven': 'saba', 'eight': 'nane', 'nine': 'tisa', 'ten': 'kumi',
            # Modifiers
            'best': 'bora', 'good': 'nzuri', 'bad': 'mbaya',
            'big': 'kubwa', 'small': 'ndogo', 'many': 'mengi',
            'all': 'yote', 'some': 'baadhi', 'few': 'chache',
        }
        
        # Create reverse mapping
        sw_to_en = {v: k for k, v in en_to_sw.items()}
        
        return {
            'en_sw': en_to_sw,
            'sw_en': sw_to_en
        }


class AdvancedTokenizer:
    """
    Advanced text tokenization with stemming and normalization.
    """
    
    def __init__(self):
        self.stop_words = self._build_stop_words()
        self.stemming_rules = self._build_stemming_rules()
        
    def tokenize(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            remove_stopwords: Whether to remove stop words
            
        Returns:
            List of tokens
        """
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        tokens = text.split()
        
        # Remove stop words if requested
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        return tokens
    
    def stem(self, word: str) -> str:
        """
        Apply stemming to reduce word to root form.
        
        Args:
            word: Word to stem
            
        Returns:
            Stemmed word
        """
        word_lower = word.lower()
        
        # Apply stemming rules
        for suffix, replacement in self.stemming_rules.items():
            if word_lower.endswith(suffix):
                return word_lower[:-len(suffix)] + replacement
        
        return word_lower
    
    def normalize(self, text: str) -> str:
        """
        Normalize text (lowercase, remove extra spaces, etc).
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation except spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Extract n-grams from text.
        
        Args:
            text: Text to process
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        tokens = self.tokenize(text)
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _build_stop_words(self) -> set:
        """Build stop words list."""
        return {
            # English
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'in', 'on', 'at', 'to', 'for', 'with', 'from', 'of',
            # Swahili
            'ni', 'wa', 'ya', 'la', 'cha', 'pa', 'ku', 'na'
        }
    
    def _build_stemming_rules(self) -> Dict[str, str]:
        """Build stemming rules (suffix -> replacement)."""
        return {
            'ing': '', 'ed': '', 'es': '', 's': '',
            'tion': 't', 'ness': '', 'ful': '', 'less': '',
            'ly': '', 'ment': '', 'ity': '', 'er': '', 'est': ''
        }
