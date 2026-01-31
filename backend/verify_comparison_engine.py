import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.getcwd())

from laravel_modules.ai_brain.central_integration_layer import CentralAI

def verify_advanced_logic():
    brain = CentralAI()
    brain.initialize()
    
    # Manually discover a tenant to avoid confidence check failures
    brain.saas.analyze_data_source("TENANT_001", {
        "tables": ["customers", "orders", "inventory", "accounting_logs"],
        "columns": ["cust_id", "tx_amt", "bal", "qty", "prod_name"]
    })
    
    # Force output to use a safe encoding or handle errors
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("\n--- TEST 1: Numeric ID & Noise Resolution (0014) ---")
    query_1 = "what is the total sales of employee 0014 ths month"
    result_1 = brain.query(query_1)
    print(f"Query: {query_1}")
    print(f"Response: {result_1.get('response')}")
    
    print("\n--- TEST 2: Multi-Employee Comparison ---")
    query_2 = "compare employee njukuibilali and shakiraismail"
    result_2 = brain.query(query_2)
    print(f"Query: {query_2}")
    print(f"Response:\n{result_2.get('response')}")
    
    print("\n--- TEST 3: Profitability Ranking ---")
    query_3 = "who is the best bring more profit to the company"
    result_3 = brain.query(query_3)
    print(f"Query: {query_3}")
    print(f"Response:\n{result_3.get('response')}")

if __name__ == "__main__":
    verify_advanced_logic()
