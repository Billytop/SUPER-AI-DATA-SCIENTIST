
import json
import random
import datetime
import os

class DataGenerator30k:
    """
    Generates 30,000+ distinct Advanced Sales & ERP Scenarios.
    Logic: Combinatorial explosion of Intents x Entities x Contexts x Lifecycle States.
    """
    
    def generate(self, count=35000):
        dataset = []
        
        # 1. Dimensions
        intents = [
            # Standard Business
            ("SALES_QUERY", ["sales", "mauzo", "revenue", "income", "mapato"]),
            ("BEST_PRODUCT", ["best product", "top item", "bidhaa bora", "inayouzika sana"]),
            ("WORST_PRODUCT", ["worst product", "slow moving", "bidhaa mbaya", "isinunuliwe"]),
            ("DEBT_CHECK", ["debt", "deni", "balance", "credit"]),
            ("RISK_ANALYSIS", ["risk", "hatari", "safe", "usalama"]),
            ("STOCK_LEVEL", ["stock", "mzigo", "inventory", "quantity"]),
            ("PROFIT_LOSS", ["profit", "faida", "loss", "hasara"]),
            
            # Advanced Order/Invoice Lifecycle (New for 30k)
            ("INVOICE_GENERATION", ["invoice", "bill", "risiti", "generate invoice", "create bill", "proforma"]),
            ("PENDING_ORDERS", ["pending order", "unshipped", "hajatumwa", "waiting sales", "bado"]),
            ("DELIVERY_STATUS", ["delivered", "imefika", "shipped", "on the way", "transit", "imepokelewa"]),
            ("RETURN_LOGIC", ["returned", "refund", "credit note", "imekurudishwa", "bad stock"]),
            ("PURCHASE_ORDER", ["po", "purchase order", "order from supplier", "agizia supplier"]),
            ("LOYALTY_PROGRAM", ["points", "loyalty", "discount points", "zawadi ya mteja"]),
        ]
        
        time_frames = [
            "today", "yesterday", "this week", "last week", "this month", "last month", 
            "this year", "last year", "Q1", "Q2", "Q3", "Q4", "Januari", "Februari", 
            "Machi", "Aprili", "Mei", "Juni", "Julai", "Agosti", "Septemba", "Oktoba", "Novemba", "Desemba"
        ]
        
        entities = [
            "John Doe", "Jane Smith", "Mangi Shop", "Mama Ntilie", "Supermarket A", "Wholesaler B",
            "Azam Juice", "Coca Cola", "Cement", "Rice 50kg", "Bakery House", "Tech Hub",
            "Onesmo Shoo", "Ali Bakari", "Fatuma Juma", "Soko Kuu", "Majengo Ltd"
        ]
        
        statuses = ["Pending", "Completed", "Delivered", "Cancelled", "Draft", "Refunded", "Partially Paid"]
        
        zones = ["Kariakoo", "Posta", "Mbezi", "Arusha", "Mwanza", "Zanzibar", "Dodoma", "Tanga", "Mbeya"]
        
        tones = ["Professional", "Urgent", "Casual", "Strict Auditor", "Strategic CSO"]
        
        # 2. Generator Loop
        print(f"Generating {count} scenarios...")
        
        for i in range(count):
            intent_code, keywords = random.choice(intents)
            time_ref = random.choice(time_frames)
            entity = random.choice(entities)
            zone = random.choice(zones)
            keyword = random.choice(keywords)
            status = random.choice(statuses)
            
            # Construct a synthetic query
            templates = [
                f"{keyword} {time_ref}",
                f"Show me {keyword} for {entity}",
                f"How many {keyword} are {status}?",
                f"Compare {keyword} {time_ref} vs previous",
                f"Generate {keyword} report for {zone}",
                f"Why is {entity} marked as {status} for {keyword}?",
                f"List all {status} {keyword} {time_ref}",
                f"Nipe ripoti ya {keyword} {entity}",
                f"{status} {keyword} imekaaje {time_ref}?",
                f"Is {entity} {status} regarding {keyword}?",
                f"Predict {status} {keyword} for {time_ref}",
                f"Audit {keyword} in {zone} for {entity}"
            ]
            
            query = random.choice(templates)
            
            # Simulated Intelligence
            risk_score = random.randint(0, 100)
            confidence = random.randint(75, 100)
            
            # Deep Reasoning Chain
            reasoning = [
                f"Intent: {intent_code}",
                f"Target: {entity} | State: {status}",
                f"Time Span: {time_ref}",
                f"Scoring: Risk {risk_score}/100",
                f"Source: Laravel ERP {intent_code.lower()} module"
            ]
            
            record = {
                "id": f"ADV-{20000+i}",
                "intent": intent_code,
                "query": query,
                "entities": {"primary": entity, "time": time_ref, "status": status, "zone": zone},
                "simulated_metrics": {"risk": risk_score, "confidence": confidence},
                "ai_reasoning": reasoning,
                "expected_domain": "Sales_Intelligence_v4"
            }
            
            dataset.append(record)
            
        return dataset

if __name__ == "__main__":
    gen = DataGenerator30k()
    data = gen.generate(35000) # Exceed 30k
    
    kb_dir = "backend/reasoning"
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)
        
    output_path = os.path.join(kb_dir, "knowledge_base_30k.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print(f"Done! Generated {len(data)} scenarios at {output_path}")
