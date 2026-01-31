import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from reasoning.agents import SQLReasoningAgent

# Comprehensive Test Suite
test_cases = [
    # Best Employee Queries
    ("mfanyakazi bora ni nani kwa mauzo", "Best Employee"),
    ("who is the best employee by sales", "Best Employee"),
    ("top employee for sales", "Best Employee"),
    
    # Best Product Queries
    ("bidhaa bora mwaka jana kwa mauzo", "Best Product"),
    ("best selling product", "Best Product"),
    ("top products", "Best Product"),
    
    # Total Sales Queries
    ("mauzo ya mwaka 2025", "Total Sales with Year Filter"),
    ("total sales yesterday", "Total Sales with Yesterday Filter"),
    ("mauzo ya jana", "Total Sales with Yesterday Filter (Swahili)"),
    ("total sales last year", "Total Sales with Last Year Filter"),
    ("what is total sales", "Total Sales (No Filter)"),
    
    # Invoice Queries
    ("how many invoices exist", "Invoice Count"),
    ("invoice count yesterday", "Invoice Count with Yesterday Filter"),
]

agent = SQLReasoningAgent()

print("="*80)
print("COMPREHENSIVE AI TEST SUITE")
print("="*80)

passed = 0
failed = 0

for query, expected_intent in test_cases:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"Expected Intent: {expected_intent}")
    print(f"{'-'*80}")
    
    result = agent.run(query)
    answer = result['answer']
    sql = result['sql']
    
    # Extract intent from answer
    if "Best Employee" in answer:
        detected_intent = "Best Employee"
    elif "Best Selling Products" in answer or "Best Product" in answer:
        detected_intent = "Best Product"
    elif "Total Sales" in answer:
        detected_intent = "Total Sales"
    elif "Count Invoices" in answer or "Invoice" in answer:
        detected_intent = "Invoice Count"
    else:
        detected_intent = "Unknown"
    
    # Check if correct
    if expected_intent.startswith(detected_intent) or detected_intent in expected_intent:
        status = "[PASS]"
        passed += 1
    else:
        status = "[FAIL]"
        failed += 1
    
    print(f"Detected: {detected_intent}")
    print(f"Status: {status}")
    print(f"SQL: {sql[:100]}...")

print(f"\n{'='*80}")
print(f"TEST SUMMARY: {passed} PASSED / {failed} FAILED / {passed+failed} TOTAL")
print(f"Success Rate: {(passed/(passed+failed)*100):.1f}%")
print(f"{'='*80}")
