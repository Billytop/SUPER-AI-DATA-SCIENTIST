
import random
from typing import List, Dict

# SOCIAL MEDIA MATRIX v3.0
# Automated Engagement Bot & Viral Trend Analyzer.

class SocialMediaMatrix:
    def __init__(self):
        self.trending_hashtags = ["#Viral", "#MustHaves", "#TechLife", "#StartupGrind", "#Innovation"]
        self.reply_bank = {
            "POSITIVE": [
                "Glad you love it! 🔥",
                "Thanks for the support! 🙏",
                "We appreciate you! 💙",
                "Legendary status unlocked. 🏆"
            ],
            "NEGATIVE": [
                "So sorry to hear this. DM us! 📩",
                "Let's fix this for you. Check your inbox.",
                "We apologize for the trouble. Team is on it. 🔧"
            ],
            "QUESTION": [
                "Great q! Check the link in bio for details. 🔗",
                "DM us for full specs!",
                "Yes! It's available now."
            ]
        }

    def analyze_comment_sentiment(self, comment: str) -> str:
        """
        Naive sentiment analysis for comment moderation.
        """
        pos_words = ["love", "great", "amazing", "best", "fire", "cool"]
        neg_words = ["bad", "worst", "broken", "hate", "slow", "fail"]
        
        comment_lower = comment.lower()
        if any(w in comment_lower for w in pos_words):
            return "POSITIVE"
        elif any(w in comment_lower for w in neg_words):
            return "NEGATIVE"
        elif "?" in comment:
            return "QUESTION"
        else:
            return "NEUTRAL"

    def auto_reply(self, comment: str) -> str:
        """
        Generates an automatic response based on sentiment.
        """
        sentiment = self.analyze_comment_sentiment(comment)
        
        if sentiment in self.reply_bank:
            return random.choice(self.reply_bank[sentiment])
        return "Thanks for commenting! 👋"

    def get_viral_hashtags(self, niche: str) -> List[str]:
        """
        Returns top hashtags for a niche.
        """
        base = self.trending_hashtags
        if niche == "TECH":
            base += ["#SoftwareDev", "#CyberSecurity", "#AI"]
        elif niche == "FASHION":
            base += ["#OOTD", "#StyleInspo", "#StreetWear"]
            
        return [f"{tag} (Reach: 50k+)" for tag in base]

SOCIAL_BOT = SocialMediaMatrix()
