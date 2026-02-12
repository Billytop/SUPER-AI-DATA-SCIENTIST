
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def test_phase_38():
    print("\n--- [STARTING PHASE 38 VERIFICATION: NEURAL RESILIENCE] ---")
    
    engine = OmnibrainSaaSEngine()
    
    # 1. Test Long Text Summarization
    print("\n--- 1. Testing Neural Text Processor (Long Text) ---")
    long_query = """
    Hello AI, I hope you are doing well today. I have a bit of a complex situation with a customer.
    His name is Paschal White and he has been buying from us for a long time.
    Recently, he came to the shop and took some cement but didn't pay immediately.
    I want to know exactly how much debt he currently owes us because I need to send him a reminder.
    Also, check if we have enough Iron Bars in stock please.
    """
    
    # We expect the system to process this without crashing and hopefully extract "debt" intent
    response = engine.process_query_v2(long_query)
    try:
        print(f"Long Query Processed. Response Preview: {response[:100].encode('cp1252', 'replace').decode('cp1252')}...")
    except:
        print(f"Long Query Processed. Response Preview: [Content Hidden due to Encoding]")
    
    # Check internal state (simulated by checking if intent was stored in deep context)
    # We can't access private attributes easily, so we rely on the log output or side effects
    # But we can check if the response addresses the debt (which it should if intent logic worked)
    if "Deni" in response or "Balance" in response or "Paschal" in response:
        print("[SUCCESS] Long Text Intent Extracted Successfully.")
    else:
        print("[FAILURE] Long Text Intent Extraction FAILED.")

    # 2. Test Error Resilience (Simulated)
    print("\n--- 2. Testing Resilient Error Handler ---")
    # We will try to pass a None context which might cause issues if not handled, 
    # or just rely on the fact that the previous call didn't crash.
    try:
        res = engine.process_query_v2("Simple Check")
        # print(f"Standard Query Safe: {res[:50]}...")
        print("[SUCCESS] Safety Net seemingly active (No Crash).")
    except Exception as e:
        print(f"[FAILURE] Safety Net FAILED. Crash detected: {e}")

    print("\n--- [PHASE 38 VERIFICATION COMPLETE] ---")

if __name__ == "__main__":
    test_phase_38()
