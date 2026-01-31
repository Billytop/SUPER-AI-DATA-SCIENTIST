"""
Explanation Engine
Generates detailed explanations for calculations, decisions, and answers.
"""

from typing import Dict, List, Optional


class ExplanationEngine:
    """
    Provides detailed explanations for why/how questions.
    """
    
    def __init__(self):
        self.calculation_explainers = self._build_calculation_explainers()
        
    def explain_calculation(self, metric: str, values: Dict) -> str:
        """Generate step-by-step calculation explanation."""
        explainer = self.calculation_explainers.get(metric.lower())
        
        if explainer:
            return explainer(values)
        
        return f"Calculation: {metric} = {values.get('result', 'N/A')}"
    
    def explain_decision(self, query: str, intent: str, confidence: float) -> str:
        """Explain how a decision was made."""
        explanation = f"## How I understood your query:\n\n"
        explanation += f"**Your question:** {query}\n\n"
        explanation += f"**Detected intent:** {intent}\n"
        explanation += f"**Confidence:** {confidence:.0%}\n\n"
        
        explanation += "##Why this intent?\n"
        explanation += self._explain_intent_selection(query, intent)
        
        return explanation
    
    def explain_result(self, query: str, result: Dict, sql: str = None) -> str:
        """Explain how a result was obtained."""
        explanation = "## How this answer was generated:\n\n"
        
        if sql:
            explanation += "### Database Query:\n```sql\n" + sql + "\n```\n\n"
        
        explanation += "### Result Processing:\n"
        explanation += f"- Found {result.get('count', 0)} matching records\n"
        explanation += f"- Applied filters based on your time period\n"
        explanation += f"- Aggregated data to provide summary\n"
        
        return explanation
    
    def generate_visual_chart(self, data: List[Dict], chart_type: str = 'bar') -> str:
        """Generate ASCII chart for visualization."""
        if not data or len(data) == 0:
            return ""
        
        if chart_type == 'bar':
            return self._create_bar_chart(data)
        elif chart_type == 'trend':
            return self._create_trend_chart(data)
        else:
            return ""
    
    def _create_bar_chart(self, data: List[Dict]) -> str:
        """Create simple ASCII bar chart."""
        if not data:
            return ""
        
        max_value = max(item.get('value', 0) for item in data)
        if max_value == 0:
            return ""
        
        chart = "\n📊 Visual Representation:\n\n"
        
        for item in data[:10]:  # Max 10 items
            name = item.get('name', 'Unknown')[:20]
            value = item.get('value', 0)
            bar_length = int((value / max_value) * 40)
            bar = "█" * bar_length
            
            chart += f"{name:<20} | {bar} {value:,.0f}\n"
        
        return chart
    
    def _create_trend_chart(self, data: List[Dict]) -> str:
        """Create trend line chart."""
        if not data or len(data) < 2:
            return ""
        
        chart = "\n📈 Trend:\n\n"
        values = [item.get('value', 0) for item in data]
        max_val = max(values)
        min_val = min(values)
        
        if max_val == min_val:
            return chart + "➡️ Flat (no change)\n"
        
        # Simple trend indicator
        if values[-1] > values[0]:
            chart += "📈 Upward trend\n"
        else:
            chart += "📉 Downward trend\n"
        
        return chart
    
    def _explain_intent_selection(self, query: str, intent: str) -> str:
        """Explain why a particular intent was selected."""
        query_lower = query.lower()
        
        explanations = {
            'SALES': "Keywords: 'sales', 'revenue', 'mauzo' detected",
            'INVENTORY': "Keywords: 'stock', 'inventory', 'mzigo' detected",
            'CUSTOMER_RISK': "Keywords: 'debt', 'outstanding', 'deni' detected",
            'COMPARISON': "Keywords: 'compare', 'vs', 'linganisha' detected",
            'BEST_PRODUCT': "Keywords: 'best', 'top' detected",
        }
        
        return explanations.get(intent, "Pattern matching and semantic analysis")
    
    def _build_calculation_explainers(self) -> Dict:
        """Build calculation explainers."""
        return {
            'gross margin': lambda v: f"""
## Gross Margin Calculation

**Formula:** ((Revenue - COGS) / Revenue) × 100

**Step 1:** Calculate Gross Profit
- Revenue: {v.get('revenue', 0):,.2f} TZS
- COGS: {v.get('cogs', 0):,.2f} TZS
- Gross Profit = {v.get('revenue', 0):,.2f} - {v.get('cogs', 0):,.2f} = {v.get('gross_profit', 0):,.2f} TZS

**Step 2:** Calculate Margin Percentage
- Gross Margin = ({v.get('gross_profit', 0):,.2f} / {v.get('revenue', 0):,.2f}) × 100
- **Result: {v.get('result', 0):.2f}%**

**Interpretation:** For every 100 TZS in sales, you keep {v.get('result', 0):.2f} TZS as gross profit.
            """,
            
            'roi': lambda v: f"""
## Return on Investment (ROI) Calculation

**Formula:** ((Gain - Cost) / Cost) × 100

**Your Investment:**
- Initial Cost: {v.get('cost', 0):,.2f} TZS
- Return/Gain: {v.get('gain', 0):,.2f} TZS
- Net Profit: {v.get('gain', 0) - v.get('cost', 0):,.2f} TZS

**Calculation:**
- ROI = ({v.get('gain', 0):,.2f} - {v.get('cost', 0):,.2f}) / {v.get('cost', 0):,.2f} × 100
- **Result: {v.get('result', 0):.2f}%**

**Interpretation:** {"Good return!" if v.get('result', 0) > 20 else "Consider improving returns."}
            """
        }
