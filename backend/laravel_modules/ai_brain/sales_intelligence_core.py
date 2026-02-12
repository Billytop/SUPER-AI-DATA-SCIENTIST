
import math
import random
from typing import Dict, List, Any

# SOVEREIGN SALES INTELLIGENCE CORE v2.0
# Advanced Logic for Dynamic Pricing, Churn Prediction, and Upselling.

class SalesIntelligenceEngine:
    def __init__(self):
        self.price_elasticity_cache = {}
        self.customer_segments = {
            "VIP": {"min_spend": 1000000, "discount": 0.10},
            "LOYAL": {"min_spend": 500000, "discount": 0.05},
            "NEW": {"min_spend": 0, "discount": 0.0}
        }

    def calculate_dynamic_price(self, base_price: float, demand_score: float, competitor_price: float) -> Dict[str, Any]:
        """
        Adjusts price based on Real-Time Demand and Competitor Analysis.
        Demand Score: 0.0 (Low) to 10.0 (High).
        """
        # Elasticity Factor: How sensitive is this item? (Mocked as 1.5 standard)
        elasticity = 1.5
        
        # 1. Demand Adjustment
        # If demand is high (>7), increase price slightly to capture value.
        # If demand is low (<3), decrease to stimulate sales.
        demand_factor = 1.0
        if demand_score > 8.0:
            demand_factor = 1.15 # +15%
        elif demand_score > 6.0:
            demand_factor = 1.05 # +5%
        elif demand_score < 3.0:
            demand_factor = 0.90 # -10%
            
        optimal_price = base_price * demand_factor
        
        # 2. Competitor Guardrail
        # Never exceed competitor price by more than 20% unless unique value proposition exists.
        if optimal_price > (competitor_price * 1.2):
            optimal_price = competitor_price * 1.2
            
        # 3. Margin Protection
        # Never sell below cost + 5% (assuming Cost is 70% of Base Price)
        min_price = base_price * 0.75
        if optimal_price < min_price:
            optimal_price = min_price
            
        return {
            "original_price": base_price,
            "optimal_price": round(optimal_price, 2),
            "demand_score": demand_score,
            "adjustment_reason": "High Demand" if demand_factor > 1 else "Stimulate Sales"
        }

    def predict_churn_risk(self, last_purchase_days_ago: int, avg_frequency: int) -> str:
        """
        Determines if a customer is about to leave.
        """
        if last_purchase_days_ago > (avg_frequency * 3):
            return "CRITICAL: CHURNED (Lost Customer)"
        elif last_purchase_days_ago > (avg_frequency * 2):
            return "HIGH RISK: Needs Reactivation Offer"
        elif last_purchase_days_ago > (avg_frequency * 1.5):
            return "MEDIUM RISK: Send Reminder"
        else:
            return "LOW RISK: Active Customer"

    def generate_next_best_offer(self, purchase_history: List[str]) -> str:
        """
        Recommends the next product based on 'Market Basket Analysis' logic.
        """
        # Simple Association Rules (Mocked logic for 50k line scale)
        rules = {
            "Laptop": "Mouse, Laptop Bag, Antivirus",
            "Phone": "Screen Protector, Case, Powerbank",
            "Cement": "Sand, Paint, Roofing Sheets",
            "Maize Flour": "Cooking Oil, Sugar, Salt",
            "Beer": "Nyama Choma, Soda, Water"
        }
        
        for item in purchase_history:
            if item in rules:
                return f"Recommended Upsell: {rules[item]}"
                
        return "Recommended Upsell: Global Top Seller (Generic)"

    def segment_customer(self, total_spend: float) -> str:
        """Classifies customer into segments."""
        if total_spend >= self.customer_segments["VIP"]["min_spend"]:
            return "VIP GOLD"
        elif total_spend >= self.customer_segments["LOYAL"]["min_spend"]:
            return "LOYAL SILVER"
        else:
            return "STANDARD BRONZE"

SALES_CORE = SalesIntelligenceEngine()
