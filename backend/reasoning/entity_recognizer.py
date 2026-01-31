"""
Advanced Entity Recognition System
Extracts and identifies entities from user queries with fuzzy matching and context awareness.
"""

from difflib import SequenceMatcher
from datetime import datetime, timedelta
import re
from django.db import connections
from typing import Dict, List, Tuple, Optional


class EntityRecognizer:
    """
    Intelligent entity extraction for products, customers, dates, numbers, and locations.
    Uses fuzzy matching and database lookups for accurate recognition.
    """
    
    def __init__(self):
        self.product_cache = []
        self.customer_cache = []
        self.cache_timestamp = None
        self.cache_ttl = 3600  # 1 hour cache
        
    def extract_entities(self, query: str) -> Dict[str, any]:
        """
        Main entry point - extracts all entity types from query.
        
        Returns dict with:
        - products: List of product names found
        - customers: List of customer names found
        - dates: List of date entities
        - numbers: List of numeric values
        - currencies: List of currency amounts
        - locations: List of locations
        """
        entities = {
            'products': self.extract_products(query),
            'customers': self.extract_customers(query),
            'dates': self.extract_dates(query),
            'numbers': self.extract_numbers(query),
            'currencies': self.extract_currencies(query),
            'locations': self.extract_locations(query)
        }
        return entities
    
    def extract_products(self, query: str, threshold: float = 0.6) -> List[Dict[str, any]]:
        """
        Extract product names using fuzzy matching against database.
        
        Args:
            query: User query string
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dicts with 'name', 'confidence', 'match_type'
        """
        self._refresh_product_cache()
        
        products_found = []
        query_lower = query.lower()
        
        # Direct exact match
        for product in self.product_cache:
            if product['name'].lower() in query_lower:
                products_found.append({
                    'name': product['name'],
                    'id': product['id'],
                    'confidence': 1.0,
                    'match_type': 'exact'
                })
        
        # Fuzzy match if no exact match
        if not products_found:
            words = query_lower.split()
            for product in self.product_cache:
                product_name = product['name'].lower()
                
                # Check each word and multi-word combinations
                for i in range(len(words)):
                    for j in range(i+1, min(i+5, len(words)+1)):  # Check up to 4-word combinations
                        phrase = ' '.join(words[i:j])
                        similarity = SequenceMatcher(None, phrase, product_name).ratio()
                        
                        if similarity >= threshold:
                            products_found.append({
                                'name': product['name'],
                                'id': product['id'],
                                'confidence': similarity,
                                'match_type': 'fuzzy',
                                'matched_phrase': phrase
                            })
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_products = []
        for p in sorted(products_found, key=lambda x: x['confidence'], reverse=True):
            if p['name'] not in seen:
                seen.add(p['name'])
                unique_products.append(p)
        
        return unique_products[:5]  # Return top 5
    
    def extract_customers(self, query: str, threshold: float = 0.7) -> List[Dict[str, any]]:
        """
        Extract customer names using fuzzy matching.
        Higher threshold for customers to avoid false positives.
        """
        self._refresh_customer_cache()
        
        customers_found = []
        query_lower = query.lower()
        
        # Exact match
        for customer in self.customer_cache:
            if customer['name'].lower() in query_lower:
                customers_found.append({
                    'name': customer['name'],
                    'id': customer['id'],
                    'confidence': 1.0,
                    'match_type': 'exact'
                })
        
        # Fuzzy match
        if not customers_found:
            words = query_lower.split()
            for customer in self.customer_cache:
                customer_name = customer['name'].lower()
                
                for i in range(len(words)):
                    for j in range(i+1, min(i+6, len(words)+1)):
                        phrase = ' '.join(words[i:j])
                        similarity = SequenceMatcher(None, phrase, customer_name).ratio()
                        
                        if similarity >= threshold:
                            customers_found.append({
                                'name': customer['name'],
                                'id': customer['id'],
                                'confidence': similarity,
                                'match_type': 'fuzzy',
                                'matched_phrase': phrase
                            })
        
        # Remove duplicates
        seen = set()
        unique_customers = []
        for c in sorted(customers_found, key=lambda x: x['confidence'], reverse=True):
            if c['name'] not in seen:
                seen.add(c['name'])
                unique_customers.append(c)
        
        return unique_customers[:3]  # Return top 3
    
    def extract_dates(self, query: str) -> List[Dict[str, any]]:
        """
        Extract date entities including relative dates, absolute dates, and ranges.
        """
        dates_found = []
        query_lower = query.lower()
        
        # Relative dates
        relative_patterns = {
            r'\b(today|leo)\b': ('today', 0),
            r'\b(yesterday|jana)\b': ('yesterday', -1),
            r'\b(tomorrow|kesho)\b': ('tomorrow', 1),
            r'\b(this\s+week|wiki\s+hii)\b': ('this_week', 0),
            r'\b(last\s+week|wiki\s+iliyopita|wiki\s+jana)\b': ('last_week', -7),
            r'\b(this\s+month|mwezi\s+huu)\b': ('this_month', 0),
            r'\b(last\s+month|mwezi\s+uliopita|mwezi\s+jana)\b': ('last_month', -30),
            r'\b(this\s+year|mwaka\s+huu)\b': ('this_year', 0),
            r'\b(last\s+year|mwaka\s+uliopita|mwaka\s+jana)\b': ('last_year', -365),
        }
        
        for pattern, (label, offset) in relative_patterns.items():
            if re.search(pattern, query_lower):
                dates_found.append({
                    'type': 'relative',
                    'label': label,
                    'offset_days': offset,
                    'confidence': 1.0
                })
        
        # Absolute dates (YYYY-MM-DD, DD/MM/YYYY, etc.)
        date_patterns = [
            (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', 'YYYY-MM-DD'),
            (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', 'DD/MM/YYYY'),
            (r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b', 'DD-MM-YYYY'),
        ]
        
        for pattern, format_type in date_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                dates_found.append({
                    'type': 'absolute',
                    'value': match.group(0),
                    'format': format_type,
                    'confidence': 1.0
                })
        
        # Number of days/months/years ago
        relative_number_patterns = [
            (r'(\d+)\s+(day|days|siku)\s+ago', 'days_ago'),
            (r'(\d+)\s+(month|months|miezi)\s+ago', 'months_ago'),
            (r'(\d+)\s+(year|years|miaka)\s+ago', 'years_ago'),
        ]
        
        for pattern, label in relative_number_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                number = int(match.group(1))
                dates_found.append({
                    'type': 'relative_number',
                    'label': label,
                    'value': number,
                    'confidence': 1.0
                })
        
        return dates_found
    
    def extract_numbers(self, query: str) -> List[Dict[str, any]]:
        """
        Extract numeric values including integers, decimals, and written numbers.
        """
        numbers_found = []
        
        # Numeric patterns
        number_patterns = [
            (r'\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\b', 'formatted_number'),  # 1,000,000.50
            (r'\b(\d+\.\d+)\b', 'decimal'),  # 123.45
            (r'\b(\d+)\b', 'integer'),  # 123
        ]
        
        for pattern, num_type in number_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                value_str = match.group(1).replace(',', '')
                try:
                    value = float(value_str)
                    numbers_found.append({
                        'type': num_type,
                        'value': value,
                        'original': match.group(1),
                        'confidence': 1.0
                    })
                except ValueError:
                    pass
        
        return numbers_found
    
    def extract_currencies(self, query: str) -> List[Dict[str, any]]:
        """
        Extract currency amounts with symbols and codes.
        """
        currencies_found = []
        
        # Currency patterns
        currency_patterns = [
            (r'(TZS|Tsh|TSH)\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?)', 'TZS'),
            (r'(TZS|Tsh|TSH)\s*(\d+(?:\.\d+)?)', 'TZS'),
            (r'(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\s*(TZS|Tsh|TSH)', 'TZS'),
            (r'(\d+(?:\.\d+)?)\s*(TZS|Tsh|TSH)', 'TZS'),
            (r'\$\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?)', 'USD'),
            (r'\$\s*(\d+(?:\.\d+)?)', 'USD'),
        ]
        
        for pattern, currency in currency_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                # Extract the numeric part
                amount_str = match.group(2) if len(match.groups()) > 1 else match.group(1)
                amount_str = amount_str.replace(',', '')
                
                try:
                    amount = float(amount_str)
                    currencies_found.append({
                        'currency': currency,
                        'amount': amount,
                        'original': match.group(0),
                        'confidence': 1.0
                    })
                except (ValueError, IndexError):
                    pass
        
        return currencies_found
    
    def extract_locations(self, query: str) -> List[Dict[str, any]]:
        """
        Extract location references (cities, regions).
        """
        locations_found = []
        query_lower = query.lower()
        
        # Tanzania cities and regions
        tanzania_locations = [
            'dar es salaam', 'dar', 'dodoma', 'mwanza', 'arusha', 'mbeya', 'morogoro',
            'tanga', 'kahama', 'tabora', 'zanzibar', 'pemba', 'mtwara', 'kigoma',
            'singida', 'iringa', 'shinyanga', 'bukoba', 'musoma', 'moshi', 'kilimanjaro'
        ]
        
        for location in tanzania_locations:
            if location in query_lower:
                locations_found.append({
                    'name': location.title(),
                    'confidence': 1.0,
                    'country': 'Tanzania'
                })
        
        return locations_found
    
    def _refresh_product_cache(self):
        """Refresh product cache if expired."""
        current_time = datetime.now().timestamp()
        
        if (not self.product_cache or 
            not self.cache_timestamp or 
            current_time - self.cache_timestamp > self.cache_ttl):
            
            with connections['erp'].cursor() as cursor:
                cursor.execute("""
                    SELECT id, name 
                    FROM products 
                    WHERE deleted_at IS NULL
                    LIMIT 1000
                """)
                rows = cursor.fetchall()
                self.product_cache = [{'id': r[0], 'name': r[1]} for r in rows]
                self.cache_timestamp = current_time
    
    def _refresh_customer_cache(self):
        """Refresh customer cache if expired."""
        current_time = datetime.now().timestamp()
        
        if (not self.customer_cache or 
            not self.cache_timestamp or 
            current_time - self.cache_timestamp > self.cache_ttl):
            
            with connections['erp'].cursor() as cursor:
                cursor.execute("""
                    SELECT id, name 
                    FROM contacts 
                    WHERE type='customer' AND deleted_at IS NULL
                    LIMIT 500
                """)
                rows = cursor.fetchall()
                self.customer_cache = [{'id': r[0], 'name': r[1]} for r in rows]
                self.cache_timestamp = current_time
