import sys
import io
import os
import time

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup Path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Import Engine
try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
except ImportError as e:
    print(f"❌ Could not import Engine: {e}")
    sys.exit(1)

def verify_predictions():
    print("--- VERIFYING PREDICTIVE ANALYTICS ---")
    sys.stdout.flush()
    
    # Initialize
    try:
        engine = OmnibrainSaaSEngine()
        print("✅ Engine Initialized.")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Engine Init Failed: {e}")
        sys.stdout.flush()
        return

    # 1. Test Sales Forecast
    print("\n1. Testing Sales Forecast...")
    sys.stdout.flush()
    try:
        response = engine._predict_future_sales()
        print(f"🔮 Forecast Output:\n---\n{response}\n---")
        sys.stdout.flush()
        
        if "Predicted Total" in response:
            print("✅ SUCCESS: Sales Forecast generated.")
        else:
            print("⚠️ WARNING: Forecast returned no data (possibly empty DB history).")
        sys.stdout.flush()
            
    except Exception as e:
        print(f"❌ Sales Forecast Failed: {e}")
        sys.stdout.flush()

    # 2. Test Stockout Prediction (using a likely product 'Cement' or 'Plug')
    # We will try to find a real product first
    print("\n2. Testing Stockout Prediction...")
    sys.stdout.flush()
    try:
        # Find a product to test
        prods = engine._execute_erp_query("SELECT name FROM products LIMIT 1")
        if prods:
            p_name = prods[0]['name']
            print(f"   Testing with product: '{p_name}'")
            sys.stdout.flush()
            response = engine._predict_stockout_date(p_name)
            print(f"📉 Stockout Output:\n---\n{response}\n---")
            sys.stdout.flush()
            
            if "Stockout Prediction" in response:
                print("✅ SUCCESS: Stockout Prediction generated.")
            else:
                 print("❌ FAILURE: Stockout Prediction logic failed.")
            sys.stdout.flush()
        else:
            print("⚠️ Skipped: No products found in DB.")
            sys.stdout.flush()

    except Exception as e:
         print(f"❌ Stockout Prediction Failed: {e}")
         sys.stdout.flush()

if __name__ == "__main__":
    verify_predictions()
