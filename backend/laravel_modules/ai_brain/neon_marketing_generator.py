
import random
from typing import List, Dict

# NEON-GENESIS MARKETING GENERATOR v1.0
# Automated Creative Engine for High-Impact Ad Copy.

class NeonMarketingAI:
    def __init__(self):
        self.power_words = {
            "HYPE": ["Explosive", "Legendary", "Unreal", "Next-Level", "Insane", "Fire", "Lit"],
            "FOMO": ["Gone in 60s", "Last Chance", "Zero Hour", "Selling Out", "Rare Drop"],
            "FORMAL": ["Premium", "Exquisite", "Professional", "Certified", "Enterprise-Grade"],
            "URGENT": ["NOW", "TODAY ONLY", "IMMEDIATE", "FLASH SALE"]
        }
        
        self.templates = {
            "INSTAGRAM": [
                "🚀 {product} just dropped! {vibe_adj} quality for the real ones. Link in bio! #{product_tag} #NeonGenesis",
                "POV: You just bought the best {product} on the market. 🔥 Don't sleep on this. {fomo_phrase}!",
                "✨ Upgrade your life with {product}. {formal_adj} design, unbeatable performance.",
            ],
            "SMS": [
                "HEY! {product} is here. {fomo_phrase}! Reply STOP to opt out.",
                "FLASH ALERT: {product} is 20% OFF. {urgent_word}. Buy now: {link}"
            ],
            "LINKEDIN": [
                "Excited to announce our new {product}. Engineered for {formal_adj} results. Let's connect.",
                "Efficiency meets Innovation. {product} is changing the game. #Business #Growth"
            ]
        }

    def generate_ad_copy(self, product_name: str, platform: str, vibe: str = "HYPE") -> str:
        """
        Generates creative text based on platform constraints and tonal vibe.
        """
        platform = platform.upper()
        vibe = vibe.upper()
        
        if platform not in self.templates:
            return "Error: Platform not supported. Use INSTAGRAM, SMS, or LINKEDIN."
            
        template = random.choice(self.templates[platform])
        
        # Fill placeholders
        ad_copy = template.format(
            product=product_name,
            product_tag=product_name.replace(" ", ""),
            vibe_adj=random.choice(self.power_words.get(vibe, self.power_words["HYPE"])),
            fomo_phrase=random.choice(self.power_words["FOMO"]),
            formal_adj=random.choice(self.power_words["FORMAL"]),
            urgent_word=random.choice(self.power_words["URGENT"]),
            link="bit.ly/shop-now"
        )
        
        return ad_copy

    def create_campaign_calendar(self, product_name: str) -> List[Dict[str, str]]:
        """
        Generates a 7-day launch strategy.
        """
        return [
            {"day": 1, "theme": "TEASER", "copy": self.generate_ad_copy(product_name, "INSTAGRAM", "FOMO")},
            {"day": 2, "theme": "REVEAL", "copy": self.generate_ad_copy(product_name, "INSTAGRAM", "HYPE")},
            {"day": 3, "theme": "FEATURES", "copy": self.generate_ad_copy(product_name, "LINKEDIN", "FORMAL")},
            {"day": 4, "theme": "SOCIAL PROOF", "copy": f"⭐⭐⭐⭐⭐ Everyone is talking about {product_name}!"},
            {"day": 5, "theme": "FOMO SPIKE", "copy": self.generate_ad_copy(product_name, "SMS", "URGENT")},
            {"day": 6, "theme": "LAST CALL", "copy": f"⚠️ {product_name} stock critical. Buy now!"},
            {"day": 7, "theme": "SOLD OUT", "copy": "We did it. Sold out. Restock soon."}
        ]

NEON_MARKETING = NeonMarketingAI()
