
import os
import sys
import logging

# Ensure the backend path is included
sys.path.append(os.path.join(os.getcwd(), "backend/laravel_modules/ai_brain"))

# Absolute imports for verification
import neon_marketing_generator
import social_media_matrix
import viral_growth_engine
from omnibrain_saas_engine import OmnibrainSaaSEngine

# Fix for Windows Unicode printing
sys.stdout.reconfigure(encoding='utf-8')

def verify_neon_genesis():
    print("\n--- [STARTING NEON-GENESIS MARKETING VERIFICATION (PHASE 33)] ---")
    
    # 1. Test Neon Marketing Generator (Ad Copy)
    print("\n--- 1. Testing Neon Marketing AI (Ad Copy) ---")
    product = "Super Sneakers"
    print(f"Generating HYPE Ad for: {product}")
    ad_copy = neon_marketing_generator.NEON_MARKETING.generate_ad_copy(product, "INSTAGRAM", "HYPE")
    print(f"Result:\n{ad_copy}")
    
    print(f"\nGenerating URGENT SMS for: {product}")
    sms_copy = neon_marketing_generator.NEON_MARKETING.generate_ad_copy(product, "SMS", "URGENT")
    print(f"Result:\n{sms_copy}")

    # 2. Test Social Media Matrix (Auto-Reply)
    print("\n--- 2. Testing Social Media Bot ---")
    comment_pos = "This product is fire! 🔥"
    print(f"Comment: '{comment_pos}'")
    reply_pos = social_media_matrix.SOCIAL_BOT.auto_reply(comment_pos)
    print(f"Bot Reply: {reply_pos}")
    
    comment_neg = "Shipping was so slow. Hate it."
    print(f"Comment: '{comment_neg}'")
    reply_neg = social_media_matrix.SOCIAL_BOT.auto_reply(comment_neg)
    print(f"Bot Reply: {reply_neg}")

    # 3. Test Viral Growth Engine
    print("\n--- 3. Testing Viral Growth Engine ---")
    invites = 12
    reward = viral_growth_engine.GROWTH_ENGINE.calculate_referral_reward(invites)
    print(f"User Invites: {invites} -> {reward}")
    
    print("Simulating Viral Growth (K-Factor 1.2):")
    growth = viral_growth_engine.GROWTH_ENGINE.simulate_viral_loop(100, 1.2)
    print(f"User Count over 10 cycles: {growth}")

    print("\n--- [NEON-GENESIS VERIFICATION COMPLETE] ---")

if __name__ == "__main__":
    verify_neon_genesis()
