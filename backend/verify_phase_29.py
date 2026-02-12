
import os
import sys
import logging

# Ensure the backend path is included
sys.path.append(os.path.join(os.getcwd(), "backend/laravel_modules/ai_brain"))

# Absolute imports for verification
import industry_experts
import market_simulator
import heuristic_matrix_ultra
import global_supply_chain
from omnibrain_saas_engine import OmnibrainSaaSEngine

def verify_mega_scale_intelligence():
    print("\n--- [STARTING MEGA-SCALE VERIFICATION (PHASE 29)] ---")
    engine = OmnibrainSaaSEngine()
    
    # 1. Test Industry Experts (Mining & Banking)
    print("\n--- 1. Testing Industry Experts ---")
    query_mining = "Analyze mining profitability for ore grade 4.2g/t"
    print(f"Query: '{query_mining}'")
    # Simulate routing to Mining Expert
    res_mining = engine._run_industry_expert_analysis("mining", query_mining)
    print(f"Result:\n{res_mining}")

    query_banking = "Check banking NPL ratio health"
    # Simulate routing to Banking Expert
    res_banking = engine._run_industry_expert_analysis("banking", query_banking)
    print(f"\nResult (Banking):\n{res_banking}")
    
    # 2. Test Market Simulator (Price War)
    print("\n--- 2. Testing Market War Room ---")
    query_sim = "Run market war simulation for competitive pricing"
    print(f"Query: '{query_sim}'")
    res_sim = engine.market_sim.run_market_war_room(query_sim)
    print(f"Result:\n{res_sim}")
    
    # 3. Test Global Supply Chain (Arbitrage)
    print("\n--- 3. Testing Global Supply Chain Arbitrage ---")
    query_arb = "Find supply chain arbitrage opportunities"
    print(f"Query: '{query_arb}'")
    res_arb = engine._run_supply_chain_optimization(query_arb)
    print(f"Result:\n{res_arb}")
    
    # 4. Test Ultra Heuristics (Specific SKU)
    print("\n--- 4. Testing Ultra-High Density Heuristics ---")
    sku = "solar panels maintenance"
    print(f"Query Item: '{sku}'")
    # Direct access for verification
    res_heuristic = engine.ultra_heuristics.get_sku_wisdom("solar", sku)
    print(f"Result:\n{res_heuristic}")

    print("\n--- [MEGA-SCALE VERIFICATION COMPLETE] ---")

if __name__ == "__main__":
    verify_mega_scale_intelligence()
