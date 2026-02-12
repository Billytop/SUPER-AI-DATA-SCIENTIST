
import os
import sys
import logging

# Ensure the backend path is included
sys.path.append(os.path.join(os.getcwd(), "backend/laravel_modules/ai_brain"))

# Absolute imports for verification
import sales_intelligence_core
import inventory_matrix_master
import hrm_neural_grid
from omnibrain_saas_engine import OmnibrainSaaSEngine

# Fix for Windows Unicode printing
sys.stdout.reconfigure(encoding='utf-8')

def verify_enterprise_core():
    print("\n--- [STARTING ENTERPRISE CORE VERIFICATION (PHASE 32)] ---")
    engine = OmnibrainSaaSEngine()
    
    # 1. Test Sales Intelligence (Dynamic Pricing)
    print("\n--- 1. Testing Sales Intelligence Core ---")
    query_price = "What is the best price for Cement?"
    print(f"Query: '{query_price}'")
    res_price = engine._run_sales_intelligence(query_price)
    print(f"Result:\n{res_price}")
    
    query_upsell = "Upsell offer for Cement purchase"
    print(f"Query: '{query_upsell}'")
    res_upsell = engine.sales_core.generate_next_best_offer(["Cement"])
    print(f"Result:\n{res_upsell}")

    # 2. Test Inventory Matrix (Shrinkage)
    print("\n--- 2. Testing Inventory Matrix Master ---")
    query_shrink = "Check for inventory shrinkage"
    print(f"Query: '{query_shrink}'")
    res_shrink = engine._run_inventory_check(query_shrink)
    print(f"Result:\n{res_shrink}")
    
    # Test Dead Stock logic directly
    print("Direct Test: Dead Stock Analysis")
    import datetime
    six_months_ago = datetime.date.today() - datetime.timedelta(days=200)
    res_dead = engine.inventory_core.analyze_stock_velocity(six_months_ago, 0.5)
    print(f"Result: {res_dead}")

    # 3. Test HRM Neural Grid (Burnout)
    print("\n--- 3. Testing HRM Neural Grid ---")
    query_hr = "Check employee burnout status"
    print(f"Query: '{query_hr}'")
    res_hr = engine._run_hrm_check(query_hr)
    print(f"Result:\n{res_hr}")
    
    # Test Performance Scoring directly
    print("Direct Test: Performance Score")
    score = engine.hrm_core.calculate_performance_score(1200000, 2, 98)
    print(f"Performance Score: {score:.2f}/100")

    print("\n--- [ENTERPRISE CORE VERIFICATION COMPLETE] ---")

if __name__ == "__main__":
    verify_enterprise_core()
