
import random
from typing import Dict, List, Any

# VIRAL GROWTH ENGINE v2.0
# Referral Loops and User Gamification Logic.

class ViralGrowthMatrix:
    def __init__(self):
        self.badges = {
            "NOVICE": "Newbie Badge 🥉",
            "ADVOCATE": "Sharer Badge 🥈",
            "INFLUENCER": "Viral King Badge 🥇",
            "WHALE": "Market Mover Badge 💎"
        }
        
        self.reward_levels = {
            5: "5000 TZS Credit",
            10: "15000 TZS Credit",
            20: "VIP Status + 50000 TZS"
        }

    def calculate_referral_reward(self, invite_count: int) -> str:
        """
        Determines the reward for a user based on invite milestones.
        """
        reward = "Keep Inviting! Next reward at 5 invites."
        
        for level, prize in sorted(self.reward_levels.items(), reverse=True):
            if invite_count >= level:
                reward = prize
                break
                
        return f"Invites: {invite_count} | Reward Unlocked: {reward}"

    def assign_user_badge(self, total_actions: int) -> str:
        """
        Gamification logic to award badges.
        """
        if total_actions > 1000:
            return self.badges["WHALE"]
        elif total_actions > 500:
            return self.badges["INFLUENCER"]
        elif total_actions > 100:
            return self.badges["ADVOCATE"]
        else:
            return self.badges["NOVICE"]

    def simulate_viral_loop(self, initial_users: int, k_factor: float) -> List[int]:
        """
        Simulates user growth over 10 cycles based on Viral Coefficient (K).
        K = Invites per user * Conversion rate.
        """
        growth = [initial_users]
        current_users = initial_users
        
        for cycle in range(10):
            new_users = int(current_users * k_factor)
            current_users += new_users
            growth.append(current_users)
            
        return growth

GROWTH_ENGINE = ViralGrowthMatrix()
