import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(r'C:\Users\njuku\Documents\AI COMPANY\SephlightyAI\backend')

try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
    brain = OmnibrainSaaSEngine()
    
    # 1. Trigger Stock Export
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
            
            # Read back using pandas to check sheets
            xl = pd.ExcelFile(filepath)
            print(f"Sheets found: {xl.sheet_names}")
            
            if 'AI_FAQS' in xl.sheet_names:
                df_faq = xl.parse('AI_FAQS')
                print(f"AI_FAQS row count: {len(df_faq)}")
                if len(df_faq) >= 30:
                    print("SUCCESS: AI_FAQS sheet has 30+ entries!")
                    sys.exit(0)
                else:
                    print(f"ERROR: AI_FAQS only has {len(df_faq)} entries.")
                    sys.exit(1)
            else:
                print("ERROR: AI_FAQS sheet missing.")
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
