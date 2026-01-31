
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Force UTF-8 output for Windows terminals
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine

def test_leaderboard(query):
    print(f"\nQUERY: {query}")
    engine = OmnibrainSaaSEngine()
    
    # We need to simulate the context since process_query expects it
    result = engine.process_query(query, "TENANT_001")
    print(f"RESPONSE:\n{result.get('response')}")
    print(f"METADATA: {result.get('metadata')}")

if __name__ == "__main__":
    # Test 1: Year 2025 ranking
    test_leaderboard("best employee of 2025")
    
    # Test 2: Swahili year 2024 ranking
    test_leaderboard("mfanyakazi bora wa 2024")
    
    # Test 3: Profit leaderboard 2025
    test_leaderboard("best profit employee 2025")
    
    # Test 4: Default month
    test_leaderboard("best employee")
