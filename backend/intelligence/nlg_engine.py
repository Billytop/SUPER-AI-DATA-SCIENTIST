"""
Natural Language Generation Engine
Generates human-friendly responses with dynamic templates and tone adjustment.
"""

from typing import Dict, List, Optional
import random


class NLGEngine:
    """
    Natural Language Generation for dynamic, context-aware responses.
    """
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self.templates = self._build_templates()
        self.tone_modifiers = self._build_tone_modifiers()
        
    def generate_response(self, intent: str, data: Dict, confidence: float, tone: str = 'professional') -> str:
        """
        Generate natural language response.
        
        Args:
            intent: Query intent
            data: Response data
            confidence: Confidence score
            tone: Response tone (professional, casual, friendly)
            
        Returns:
            Generated response text
        """
        # Select template based on intent and confidence
        template = self._select_template(intent, confidence)
        
        # Fill template with data
        response = self._fill_template(template, data)
        
        # Apply tone modifications
        response = self._apply_tone(response, tone)
        
        return response
    
    def format_number(self, number: float, format_type: str = 'currency') -> str:
        """Format numbers for display."""
        if format_type == 'currency':
            return f"{number:,.2f} TZS"
        elif format_type == 'percentage':
            return f"{number:.1f}%"
        elif format_type == 'count':
            return f"{int(number):,}"
        else:
            return f"{number:,.2f}"
    
    def format_comparison(self, current: float, previous: float, metric: str) -> str:
        """Generate comparison text."""
        change = current - previous
        percent_change = (change / previous * 100) if previous != 0 else 0
        
        if change > 0:
            direction = "increased" if self.language == 'en' else "imeongezeka"
            emoji = "📈"
        elif change < 0:
            direction = "decreased" if self.language == 'en' else "imepungua"
            emoji = "📉"
        else:
            direction = "remained same" if self.language == 'en' else "imebaki sawa"
            emoji = "➡️"
        
        return f"{emoji} {metric.title()} {direction} by {abs(percent_change):.1f}% ({self.format_number(abs(change))})"
    
    def generate_insights(self, data: Dict, intent: str) -> List[str]:
        """Generate actionable insights from data."""
        insights = []
        
        if intent == 'SALES':
            if data.get('trend') == 'increasing':
                insights.append("📈 Sales are trending upward - great momentum!")
            elif data.get('trend') == 'decreasing':
                insights.append("⚠️ Sales declining - investigate causes and take action")
            
            if data.get('best_day'):
                insights.append(f"💡 {data['best_day']} is your strongest sales day")
        
        elif intent == 'INVENTORY':
            low_stock_count = data.get('low_stock_count', 0)
            if low_stock_count > 0:
                insights.append(f"⚠️ {low_stock_count} items below reorder point - order soon")
        
        elif intent == 'CUSTOMER_RISK':
            high_risk_count = data.get('high_risk_count', 0)
            if high_risk_count > 0:
                insights.append(f"🚨 {high_risk_count} customers with debt > 60 days - urgent follow-up needed")
        
        return insights
    
    def _select_template(self, intent: str, confidence: float) -> Dict:
        """Select appropriate template based on intent and confidence."""
        intent_templates = self.templates.get(intent, self.templates['UNKNOWN'])
        
        if confidence > 0.8:
            return random.choice(intent_templates['high_confidence'])
        elif confidence > 0.6:
            return random.choice(intent_templates['medium_confidence'])
        else:
            return random.choice(intent_templates['low_confidence'])
    
    def _fill_template(self, template: str, data: Dict) -> str:
        """Fill template with actual data."""
        try:
            return template.format(**data)
        except KeyError:
            return template
    
    def _apply_tone(self, response: str, tone: str) -> str:
        """Apply tone modifications to response."""
        modifiers = self.tone_modifiers.get(tone, {})
        
        for formal, replacement in modifiers.items():
            response = response.replace(formal, replacement)
        
        return response
    
    def _build_templates(self) -> Dict:
        """Build response templates."""
        return {
            'SALES': {
                'high_confidence': [
                    "📊 Total sales: {total}. Based on {count} transactions. {insights}",
                    "💰 Sales performance: {total} from {count} orders. {insights}"
                ],
                'medium_confidence': [
                    "📊 Sales data shows: {total}. Please verify if this matches your expectations."
                ],
                'low_confidence': [
                    "I found sales data: {total}, but I'm not entirely certain this matches your query. Could you clarify?"
                ]
            },
            'INVENTORY': {
                'high_confidence': [
                    "📦 Total inventory value: {total}. {count} product types in stock. {insights}",
                    "🏪 Stock summary: {total} across {count} items. {insights}"
                ],
                'medium_confidence': [
                    "📦 inventory shows: {total}. Please verify this is what you're looking for."
                ],
                'low_confidence': [
                    "Found inventory data, but need clarification on your exact needs."
                ]
            },
            'UNKNOWN': {
                'high_confidence': ["{result}"],
                'medium_confidence': ["{result}"],
                'low_confidence': ["I'm not certain about this query. Could you rephrase?"]
            }
        }
    
    def _build_tone_modifiers(self) -> Dict:
        """Build tone modification mappings."""
        return {
            'casual': {
                'Please': 'Hey',
                'Total': 'You got',
                'Based on': 'From',
            },
            'friendly': {
                'Total': 'Your total',
                'shows': 'looks like',
            },
            'professional': {}  # No modifications for professional tone
        }
