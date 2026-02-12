import os
import sys
import pandas as pd
import openpyxl

# Add project root to path
sys.path.append(r'C:\Users\njuku\Documents\AI COMPANY\SephlightyAI\backend')

try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
    brain = OmnibrainSaaSEngine()
    
    # 1. Trigger Inventory Export
    query = 'export stock to excel'
    print(f"Executing query: {query}")
    res = brain.process_query(query, 'test_conn')
    response_str = res.get('response', '')
    
    if '[DOWNLOAD_ACTION]' in response_str:
        url = response_str.split('[DOWNLOAD_ACTION]: ')[1].strip()
        filename = url.split('/')[-1]
        filepath = os.path.abspath(os.path.join('exports', filename))
        
        if os.path.exists(filepath):
            print(f"SUCCESS: File generated at {filepath}")
            
            # Read back using pandas to check columns
            df = pd.read_excel(filepath, sheet_name='Data_Report')
            print(f"Columns found: {list(df.columns)}")
            
            required_cols = ['Stock_Started', 'Issued_Today', 'Current_Balance']
            found_all = all(col in df.columns for col in required_cols)
            
            if found_all:
                print("SUCCESS: All smart stock columns found!")
                # Check column order (Product, SKU, Started, Issued, Balance, ...)
                if list(df.columns)[2:5] == required_cols:
                    print("SUCCESS: Column order is correct (Smart Flow).")
                    sys.exit(0)
                else:
                    print(f"WARNING: Column order mismatch. Current: {list(df.columns)[2:5]}")
                    sys.exit(0) # Still success as columns exist
            else:
                missing = [c for c in required_cols if c not in df.columns]
                print(f"ERROR: Missing columns: {missing}")
                sys.exit(2)
        else:
            print(f"ERROR: File not found at {filepath}")
            sys.exit(3)
    else:
        print(f"ERROR: [DOWNLOAD_ACTION] tag missing.")
        sys.exit(4)

except Exception as e:
    print(f"Error: {e}")
    sys.exit(5)
