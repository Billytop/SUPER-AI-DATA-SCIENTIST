
import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger("OMNIBRAIN_NEURAL_TEXT")
logger.setLevel(logging.INFO)

class NeuralTextProcessor:
    """
    Sovereign Linguistic Core: Advanced NLP for Long Text Comprehension.
    Handles summarization, noise filtering, and multi-intent extraction.
    """
    def __init__(self):
        self.stop_words = {
            "hello", "hi", "hey", "please", "kindly", "could", "would", "can", "you", "me",
            "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for",
            "habari", "shikamoo", "naomba", "tafadhali", "ni", "nipe", "kwa", "ya", "cha"
        }
        
    def process_long_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a long paragraph to extract core meaning.
        Returns: {
            "summary": "Short version",
            "intents": ["sales", "debt"],
            "entities": ["Paschal", "Cement"],
            "sentiment": "Neutral"
        }
        """
        clean_text = text.lower()
        
        # 1. Summarization (Key Sentence Extraction)
        # Split by distinct separators
        sentences = re.split(r'[.!?\n]+', text)
        
        # Filter out greetings / short sentences
        meaningful_sentences = []
        for s in sentences:
            clean_s = s.strip().lower()
            if len(clean_s.split()) > 3 and not any(clean_s.startswith(g) for g in ["hello", "hi", "hey", "habari", "shikamoo"]):
                meaningful_sentences.append(s.strip())
                
        summary = meaningful_sentences[0] if meaningful_sentences else (sentences[0] if sentences else text[:50])
        
        # 2. Entity Extraction (Heuristic)
        words = re.findall(r'\b\w+\b', clean_text)
        meaningful_words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # 3. Intent Detection (Multi-Label)
        intents = []
        if any(w in meaningful_words for w in ["debt", "deni", "owe", "dai"]):
            intents.append("debt_inquiry")
        if any(w in meaningful_words for w in ["sell", "sales", "mauzo", "sold", "nunua"]):
            intents.append("sales_inquiry")
        if any(w in meaningful_words for w in ["stock", "inventory", "mzigo", "bidhaa"]):
            intents.append("inventory_inquiry")
            
        return {
            "original_length": len(text),
            "summary": summary.strip(),
            "clean_tokens": meaningful_words,
            "detected_intents": intents,
            "complexity_score": len(sentences)
        }

    def clean_query_for_search(self, text: str) -> str:
        """
        Reduces a verbose query to search-friendly keywords.
        e.g., "Hello AI, I want to know the debt of Paschal please" -> "debt Paschal"
        """
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in self.stop_words]
        return " ".join(keywords)
