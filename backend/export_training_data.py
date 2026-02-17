import os
import sys
import json
import random
import datetime
import io

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup Path to import backend modules
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Use the OmniBrain Engine to access data safely
try:
    from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
except ImportError as e:
    print(f"CRITICAL: Could not import OmniBrain Engine. {e}")
    sys.exit(1)

def generate_training_data():
    print("--- STARTING HUGE DATA EXPORT FOR AI TRAINING ---")
    engine = OmnibrainSaaSEngine()
    
    dataset = []

    # 1. PRODUCT KNOWLEDGE (The "Inventory Brain")
    print("🧠 Extracting Product Knowledge...")
    products = engine._execute_erp_query("SELECT name, sku, alert_quantity FROM products LIMIT 500")
    if products:
        for p in products:
            # Q&A Pair 1: SKU Lookup
            dataset.append({
                "instruction": f"What is the SKU for the product '{p['name']}'?",
                "input": "",
                "output": f"The SKU for '{p['name']}' is {p['sku']}."
            })
            # Q&A Pair 2: Stock Alert Logic (Reasoning)
            dataset.append({
                "instruction": "Analyze the stock alert level for this product.",
                "input": f"Product: {p['name']}, Alert Qty: {p['alert_quantity']}",
                "output": f"The product '{p['name']}' has a reorder point set at {p['alert_quantity']} units. If stock falls below this, a purchase order should be raised."
            })

    # 2. CUSTOMER INTELLIGENCE (The "Sales Brain")
    print("🧠 Extracting Customer Intelligence...")
    customers = engine._execute_erp_query("SELECT id, name, mobile FROM contacts WHERE type='customer' LIMIT 200")
    if customers:
        for c in customers:
            # Get Debt
            debt_res = engine._resolve_contact_debt(c)
            # Chain of Thought Training Example
            dataset.append({
                "instruction": f"Analyze the financial status for customer {c['name']}.",
                "input": "",
                "output": f"Thinking Process:\n1. Checked customer ID {c['id']}.\n2. Calculated total sales vs total payments.\n3. Result: {debt_res}"
            })

    # 3. TRANSACTION HISTORY (The "Ledger Brain")
    print("🧠 Extracting Transaction Patterns...")
    # Get synthesis of sales
    sales_data = engine._execute_erp_query("SELECT transaction_date, final_total FROM transactions WHERE type='sell' ORDER BY transaction_date DESC LIMIT 100")
    if sales_data:
        for s in sales_data:
            dt = s['transaction_date'].strftime('%Y-%m-%d')
            amount = float(s['final_total'])
            dataset.append({
                "instruction": f"What happened on {dt}?",
                "input": "Sales Data",
                "output": f"On {dt}, a significant sale of {amount:,.0f} TZS was recorded. This contributes to the daily revenue revenue stream."
            })

    # 4. CODEBASE UNDERSTANDING (Self-Reflection)
    # The user wants it to "interact with codes too well"
    dataset.append({
        "instruction": "How does the OmniBrain _resolve_business_data method work?",
        "input": "Source Code Analysis",
        "output": "The _resolve_business_data method acts as a router. It first cleans the query using regex, checks for specific intents like 'sales', 'debt', or 'inventory', and then dispatches the request to specialized handlers like _resolve_contact_debt or the VisualizationEngine."
    })

    # Save to JSONL
    output_file = "sephlighty_training_data.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"\n✅ SUCCESS: Exported {len(dataset)} reasoning examples to {output_file}")
    print("🚀 Ready for GPU Training!")

if __name__ == "__main__":
    generate_training_data()
