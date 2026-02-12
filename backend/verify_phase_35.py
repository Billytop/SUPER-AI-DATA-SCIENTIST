
import os
import sys

# Ensure the backend path is included
sys.path.append(os.path.join(os.getcwd(), "backend/laravel_modules/ai_brain"))

# Absolute imports for verification
import large_context_analyzer
import generative_report_engine
import sovereign_cortex_brain
from omnibrain_saas_engine import OmnibrainSaaSEngine

# Fix for Windows Unicode printing
sys.stdout.reconfigure(encoding='utf-8')

def verify_sovereign_cortex():
    print("\n--- [STARTING SOVEREIGN CORTEX VERIFICATION (PHASE 35)] ---")
    
    # 1. Test Large Context Analyzer (Simulated 10k token doc)
    print("\n--- 1. Processing Large Document (10k+ context sim) ---")
    long_text = "The company reported a massive revenue surge of 15% due to strategic tax compliance in Angola. However, operational risks remain high in the construction sector." * 50
    analysis = large_context_analyzer.CONTEXT_ANALYZER.process_large_document(long_text)
    print(f"Document Size: {analysis['document_length']} chars")
    print(f"Chunks Processed: {analysis['total_chunks']}")
    print(f"Key Insight: {analysis['detailed_insights'][0]['key_point']}")

    # 2. Test Generative Report Engine
    print("\n--- 2. Generating Executive Summary ---")
    context_data = {
        "title": "Project Alpha: Luanda Skyscraper",
        "goal": "Build 60-floor tower with optimized tax exposure",
        "financial_outlook": "Positive (ROI > 25%)",
        "risk_level": "LOW",
        "analysis_text": "Our AI has determined that local sourcing of steel combined with WHT incentives will maximize profit.",
        "rec_1": "Proceed with construction immediately.",
        "rec_2": "Lock in cement prices.",
        "rec_3": "Review labor laws."
    }
    report = generative_report_engine.REPORT_GENERATOR.generate_executive_summary(context_data)
    print(report)

    # 3. Test Sovereign Cortex Brain (Genius Reasoning)
    print("\n--- 3. Testing Genius Reasoning Engine ---")
    query = "Analyze the feasibility of building a 60-floor tower in Luanda with tax implications."
    print(f"Complex Query: '{query}'")
    synthesis = sovereign_cortex_brain.CORTEX_BRAIN.generate_genius_response(query, {})
    print(synthesis)

    print("\n--- [SOVEREIGN CORTEX VERIFICATION COMPLETE] ---")

if __name__ == "__main__":
    verify_sovereign_cortex()
