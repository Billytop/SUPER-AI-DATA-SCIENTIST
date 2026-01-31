import spacy
from textblob import TextBlob
import re

class NaturalLanguageProcessor:
    """
    Handles linguistic analysis of user queries.
    """
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if model not downloaded yet
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text):
        """
        Extracts Named Entities (Dates, Orgs, Products).
        """
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            entities[ent.label_] = ent.text
            
        return entities

    def detect_intent(self, text):
        """
        Classifies intent based on keywords and grammar.
        """
        text_lower = text.lower()
        if any(word in text_lower for word in ['predict', 'forecast', 'future', 'will']):
            return 'FORECAST'
        if any(word in text_lower for word in ['compare', 'vs', 'difference']):
            return 'COMPARISON'
        if any(word in text_lower for word in ['why', 'cause', 'reason']):
            return 'EXPLANATION'
            
        return 'QUERY_DATA'

    def clean_query(self, text):
        """
        Removes stopwords and extracts key search terms.
        Handles Swahili/Slang translation.
        """
        # Dictionary-based translation for common terms
        replacements = {
            "mauzo": "sales",
            "jana": "yesterday",
            "pesa": "money",
            "zote": "all",
            "tym": "time",
            "btwn": "between",
            "ziko": "are",
            "wapi": "where",
            "niambie": "tell me",
            "mfanyakazi": "employee",
            "bora": "best",
            "nani": "who",
            "mwaka": "year",
        }
        
        words = text.lower().split()
        translated_words = [replacements.get(w, w) for w in words]
        text = " ".join(translated_words)

        doc = self.nlp(text)
        # Keep Nouns, Verbs, Adjectives, Proper Nouns, Numbers
        tokens = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM']]
        return " ".join(tokens)
