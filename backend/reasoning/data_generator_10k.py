
import json
import random
import datetime

class DataGenerator10k:
    """
    Generates 10,000+ distinct Customer Intelligence Scenarios.
    Logic: Combinatorial explosion of Intents x Entities x Contexts x Tones.
    """
    
    def generate(self, count=12000):
        dataset = []
        
        # 1. Dimensions
        intents = [
            # Sales
            ("SALES_QUERY", ["sales", "mauzo", "revenue", "income", "mapato"]),
            ("BEST_PRODUCT", ["best product", "top item", "bidhaa bora", "inayouzika sana"]),
            ("WORST_PRODUCT", ["worst product", "slow moving", "bidhaa mbaya", "isinunuliwe"]),
            # Debt / Risk
            ("DEBT_CHECK", ["debt", "deni", "balance", "credit"]),
            ("RISK_ANALYSIS", ["risk", "hatari", "safe", "usalama"]),
            ("PAYMENT_HISTORY", ["payment", "malipo", "paid", "amelipa"]),
            # Inventory
            ("STOCK_LEVEL", ["stock", "mzigo", "inventory", "quantity"]),
            ("REORDER_POINT", ["reorder", "agiza", "isha", "low stock"]),
            # HRM
            ("EMPLOYEE_PERF", ["performance", "utendaji", "sales by", "mauzo ya"]),
            ("THEFT_DETECT", ["voids", "wizi", "cancelled", "futwa"]),
            # Tax
            ("VAT_CALC", ["vat", "kodi", "tax", "tra"]),
            ("PROFIT_LOSS", ["profit", "faida", "loss", "hasara"]),
        ]
        
        time_frames = [
            "today", "yesterday", "this week", "last week", "this month", "last month", 
            "this year", "last year", "2024", "2025", "Q1", "Q4", "Januari", "December"
        ]
        
        entities = [
            "John Doe", "Jane Smith", "Mangi Shop", "Mama Ntilie", "Supermarket A", "Wholesaler B",
            "Azam Juice", "Coca Cola", "Cement", "Rice 50kg", "Sugar", "Soap"
        ]
        
        zones = ["Kariakoo", "Posta", "Mbezi", "Arusha", "Mwanza", "Zanzibar"]
        
        tones = ["Professional", "Urgent", "Casual (Sheng)", "Strict Auditor"]
        
        # 2. Generator Loop
        print(f"Generating {count} scenarios...")
        
        for i in range(count):
            intent_code, keywords = random.choice(intents)
            time_ref = random.choice(time_frames)
            entity = random.choice(entities)
            zone = random.choice(zones)
            keyword = random.choice(keywords)
            
            # Construct a synthetic query
            templates = [
                f"{keyword} {time_ref}",
                f"Show me {keyword} for {entity}",
                f"Compare {keyword} {time_ref} vs last year",
                f"Why is {entity} {keyword} so low {time_ref}?",
                f"Analyze {keyword} at {zone}",
                f"Nipe ripoti ya {keyword} {entity}",
                f"{keyword} imekaaje {time_ref}?",
                f"Is {entity} safe regarding {keyword}?",
                f"Predict {keyword} for {time_ref}",
                f"Audit {keyword} records for {entity}"
            ]
            
            query = random.choice(templates)
            
            # Simulated Insight/Reasoning
            risk_score = random.randint(0, 100)
            confidence = random.randint(70, 99)
            
            # Deep Reasoning Chain (The "Thought Process")
            reasoning = [
                f"Detected Intent: {intent_code}",
                f"Extracted Entities: {entity}, {time_ref}",
                f"Context: Localized to {zone}",
                f"Risk Assessment: {risk_score}/100",
                f"Action: Querying {intent_code} logic..."
            ]
            
            record = {
                "id": f"CI-{10000+i}",
                "intent": intent_code,
                "query": query,
                "entities": {"primary": entity, "time": time_ref, "location": zone},
                "simulated_risk": risk_score,
                "ai_reasoning": reasoning,
                "expected_sql_template": f"SELECT * FROM transactions WHERE type='{intent_code}' AND date='{time_ref}'...",
                "training_confidence": confidence
            }
            
            dataset.append(record)
            
        return dataset

if __name__ == "__main__":
    gen = DataGenerator10k()
    data = gen.generate(12500) # Generate 12.5k to be safe
    
    with open("backend/reasoning/knowledge_base_10k.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print("Done! Generated 12,500 scenarios.")
