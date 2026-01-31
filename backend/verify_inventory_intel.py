import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.getcwd())

from laravel_modules.ai_brain.central_integration_layer import CentralAI

def verify_inventory():
    brain = CentralAI()
    brain.initialize()
    
    print("\n--- TEST 1: Fuzzy Product & Stock Movement ---")
    query = "stock movement of vin tape black"
    result = brain.query(query)
    print(f"Query: {query}")
    print(f"Response:\n{result.get('response')}")
    print(f"Metadata: {result.get('metadata')}")
    
    if "VIN TAPE BLACK" in result.get('response') and "MAUZO" in result.get('response'):
        print("\nVERIFICATION SUCCESS: Product identified and stock movement retrieved.")
    else:
        print("\nVERIFICATION FAILURE: Product or movement data missing.")

if __name__ == "__main__":
    verify_inventory()
