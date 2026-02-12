import sys
import os
import datetime

# Add the brain directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'laravel_modules/ai_brain')))

from omnibrain_saas_engine import OmnibrainSaaSEngine

def test_temporal_universal():
    brain = OmnibrainSaaSEngine()
    last_year = datetime.datetime.now().year - 1
    
    # Test Cases
    tests = [
        {
            "name": "Historical Sales (mwaka jan shorthand)",
            "query": "mauzo ya decembe mwaka jan",
            "expect": f"Disemba {last_year}"
        },
        {
            "name": "Specific Day Historical",
            "query": "mauzo ya december tarehe 21 mwaka jan",
            "expect": f"21 Disemba {last_year}"
        },
        {
            "name": "Historical Debt",
            "query": "deni la Paschal mwezi wa 12 mwaka jan",
            "expect": f"mwezi 12/{last_year}"
        },
        {
            "name": "Historical Inventory Valuation",
            "query": "thamani ya stoo mwezi wa 12 mwaka jan",
            "expect": f"MWEZI 12/{last_year}"
        },
        {
            "name": "Historical Leaderboard",
            "query": "wateja bora mwezi wa 12 mwaka jan",
            "expect": f"DEC {last_year}"
        }
    ]
    
    success_count = 0
    for test in tests:
        print(f"\n[TEST]: {test['name']}")
        print(f"Query: {test['query']}")
        res = brain.process_query(test['query'], 'test_conn')
        response = res.get('response', '')
        print(f"Response: {response}")
        
        if test['expect'] in response:
            print(f"SUCCESS: Found '{test['expect']}' in response.")
            success_count += 1
        else:
            print(f"FAILED: Could not find '{test['expect']}' in response.")
            
    print(f"\nFinal Result: {success_count}/{len(tests)} tests passed.")
    if success_count < len(tests):
        sys.exit(1)

if __name__ == "__main__":
    test_temporal_universal()
