
import os
import sys
import logging

# Ensure the backend path is included
sys.path.append(os.path.join(os.getcwd(), "backend/laravel_modules/ai_brain"))

# Absolute imports for verification
import neural_network_core
import lstm_business_engine
import nlp_advanced_processor
import knowledge_base_expansion
from omnibrain_saas_engine import OmnibrainSaaSEngine

# Fix for Windows Unicode printing
sys.stdout.reconfigure(encoding='utf-8')

def verify_neural_galaxy():
    print("\n--- [STARTING NEURAL GALAXY VERIFICATION (PHASE 30)] ---")
    engine = OmnibrainSaaSEngine()
    
    # 1. Test LSTM Business Engine (Forecasting)
    print("\n--- 1. Testing LSTM Business Engine ---")
    query_forecast = "Tabiri mauzo ya kesho (Predict revenue)"
    print(f"Query: '{query_forecast}'")
    res_forecast = engine._run_lstm_forecast()
    print(f"Result:\n{res_forecast}")

    # 2. Test Sovereign NLP (Sentiment Analysis)
    print("\n--- 2. Testing Sovereign NLP ---")
    query_sentiment = "Biashara inaenda vizuri sana, faida imeongezeka!"
    print(f"Query: '{query_sentiment}'")
    res_sentiment = engine._run_sentiment_analysis(query_sentiment)
    print(f"Result:\n{res_sentiment}")
    
    query_negative = "Hali ni mbaya, hasara tupu na madeni yamezidi."
    print(f"Query: '{query_negative}'")
    res_neg = engine._run_sentiment_analysis(query_negative)
    print(f"Result:\n{res_neg}")
    
    # 3. Test Knowledge Base (Tax Law)
    print("\n--- 3. Testing Sovereign Knowledge Base ---")
    query_law = "Nipe sheria ya kodi ya TRA (Tax Law)"
    print(f"Query: '{query_law}'")
    res_law = engine._query_knowledge_base(query_law)
    print(f"Result:\n{res_law}")

    # 4. Test Neural Core (Network Creation)
    print("\n--- 4. Testing Neural Network Core ---")
    nn = neural_network_core.create_business_brain()
    sample_input = [0.5] * 10 
    prediction = nn.predict(sample_input)
    print(f"Neural Network Output (Profit Score): {prediction[0]:.4f}")

    print("\n--- [NEURAL GALAXY VERIFICATION COMPLETE] ---")

if __name__ == "__main__":
    verify_neural_galaxy()
