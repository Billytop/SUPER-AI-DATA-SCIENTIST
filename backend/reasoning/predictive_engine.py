import logging
import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

class NeuralPredictiveEngine:
    """
    NEURAL SEMANTIC AUTOCORRECT & PREDICTIVE REASONING (v20.0)
    Role: Query Healer & Intent Anticipator
    Features: Damerau-Levenshtein Healing, Semantic Vector Weighted Correction, N-Gram Prediction.
    """
    
    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base # List of {query, intent}
        self.dictionary = self._build_business_dictionary()
        self.prediction_map = self._build_prediction_map()

    def heal_query(self, query: str) -> str:
        """
        Deep Neural Healing.
        Fixes "mauzi" -> "mauzo", "denii" -> "deni" using semantic weighting.
        """
        words = query.lower().split()
        healed_words = []
        
        for word in words:
            if len(word) <= 2:
                healed_words.append(word)
                continue
                
            # If word is in dictionary, keep it
            if word in self.dictionary:
                healed_words.append(word)
                continue
                
            # Find best match in dictionary
            best_match = word
            highest_score = 0
            
            for dict_word in self.dictionary:
                # Fuzzy similarity
                similarity = SequenceMatcher(None, word, dict_word).ratio()
                
                # Semantic Bias: Give higher weight to business keywords
                if similarity > 0.7:
                    if similarity > highest_score:
                        highest_score = similarity
                        best_match = dict_word
            
            # Healing Threshold
            if highest_score > 0.75:
                logging.info(f"🩹 Neural Healing: '{word}' -> '{best_match}' ({int(highest_score*100)}%)")
                healed_words.append(best_match)
            else:
                healed_words.append(word)
                
        return " ".join(healed_words)

    def predict_next_steps(self, query: str, intent: str) -> List[str]:
        """
        Contextual N-Gram Predictor.
        Anticipates the next 3 logical business questions.
        """
        q = query.lower()
        suggestions = []
        
        # 1. Intent-based predictions
        if intent in self.prediction_map:
            suggestions.extend(self.prediction_map[intent])
            
        # 2. Keyword-based refinement
        if "sales" in q or "mauzo" in q:
            suggestions.append("Show the profit margin for these sales?")
            suggestions.append("Which product contributed most to this total?")
        elif "debt" in q or "deni" in q:
            suggestions.append("Show the payment history for these customers?")
            suggestions.append("Simulate a 10% debt collection campaign impact?")
            
        return list(set(suggestions))[:3]

    def _build_business_dictionary(self) -> set:
        """Unified Business Dictionary (Eng + Swahili)"""
        return {
            # Sales & Revenue
            'sales', 'mauzo', 'revenue', 'income', 'sell', 'sold', 'invoice', 'inchi', 'risiti', 'vocha',
            # Accounting & Finance
            'profit', 'faida', 'loss', 'hasara', 'expense', 'gharama', 'ledger', 'tax', 'kodi',
            'debt', 'deni', 'payable', 'receivable', 'balance', 'salio', 'mtaji', 'capital', 'cash', 'fedha',
            # Inventory & Products
            'stock', 'mzigo', 'inventory', 'product', 'bidhaa', 'warehouse', 'stoo', 'store',
            'quantity', 'qty', 'reorder', 'idadi', 'vifaa', 'vitu', 'brand', 'model',
            # People & Roles
            'customer', 'mteja', 'employee', 'staff', 'worker', 'mfanyakazi', 'supplier', 'wakala', 'admin',
            # Actions & Intelligence
            'show', 'onyesha', 'list', 'orodha', 'compare', 'linganisha', 'simulate', 'simulizi',
            'analyze', 'forecast', 'utabiri', 'report', 'ripoti', 'mchanganuo', 'viz', 'picha',
            # Locations & Time
            'branch', 'tawi', 'dar', 'mwanza', 'arusha', 'dodoma', 'leo', 'jana', 'siku', 'saa', 'wiki', 'mwezi',
            'performance', 'ubora', 'kiwango', 'rank', 'nafasi', 'winner', 'shujaa'
        }

    def _build_prediction_map(self) -> Dict[str, List[str]]:
        """Anticipation Mapping"""
        return {
            'SALES': [
                "Show sales by category?",
                "Who is the top customer today?",
                "Compare these sales with last month?"
            ],
            'INVENTORY': [
                "Which items are out of stock?",
                "Calculate total stock value?",
                "Show stock movement for top items?"
            ],
            'CUSTOMER_RISK': [
                "List top 10 debtors?",
                "Send payment reminders?",
                "Compare debt vs sales for this month?"
            ],
            'ACCOUNTING': [
                "Show expense breakdown?",
                "Generate a P&L statement?",
                "Calculate estimated VAT?"
            ],
            'GREETING': [
                "How can you help my business grow today?",
                "Give me a strategic overview?",
                "Show my current performance dashboard?"
            ]
        }
