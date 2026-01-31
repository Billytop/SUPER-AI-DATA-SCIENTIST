PHASE_A_QUESTIONS = [
    ("What is total sales today?", "SELECT SUM(final_total) FROM sales_transaction WHERE transaction_date = CURRENT_DATE()"),
    ("How many invoices exist?", "SELECT COUNT(*) FROM sales_transaction"),
    ("Which table stores payments?", "TransactionPayment"), # Conceptual Answer
]

PHASE_B_LOGIC_CHECKS = [
    "CHECK_INVOICE_TOTALS", # Verify line items sum to total
    "CHECK_STOCK_DEDUCTION", # Verify stock moves match sales
]

PHASE_D_NLP_TESTS = [
    ("Niambie mauzo ya jana", "sale"), # Swahili mix
    ("how much did we sell last tym?", "sell"), # Typos
    ("pesa zote ziko wapi?", "money"), 
    ("show me sales btwn jan and feb", "sale"), 
]
