from django.db import connections
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def audit_stock_linkage():
    cursor = connections['erp'].cursor()
    
    print("--- VARIATIONS SCHEMA ---")
    try:
        cursor.execute("DESCRIBE variations")
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]}")
    except:
        print("Table 'variations' not found.")

    print("\n--- TRANSACTION_SELL_LINES SCHEMA ---")
    try:
        cursor.execute("DESCRIBE transaction_sell_lines")
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]}")
    except:
        print("Table 'transaction_sell_lines' not found.")

    print("\n--- PURCHASE_LINES SCHEMA ---")
    try:
        cursor.execute("DESCRIBE purchase_lines")
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]}")
    except:
        print("Table 'purchase_lines' not found.")

    print("\n--- RECENT STOCK MOVEMENTS FOR 'VIN TAPE BLACK' (ID: 1039) ---")
    # First find variation_id for product 1039
    cursor.execute("SELECT id FROM variations WHERE product_id = 1039")
    variation_ids = [row[0] for row in cursor.fetchall()]
    print(f"Variation IDs: {variation_ids}")
    
    if variation_ids:
        v_id = variation_ids[0]
        # Check sales
        print("\n[Sales Records]")
        cursor.execute("""
            SELECT t.id, t.ref_no, t.type, t.transaction_date, l.quantity 
            FROM transactions t
            JOIN transaction_sell_lines l ON t.id = l.transaction_id
            WHERE l.variation_id = %s
            ORDER BY t.transaction_date DESC LIMIT 5
        """, (v_id,))
        for row in cursor.fetchall():
            print(row)
            
        # Check purchases
        print("\n[Purchase Records]")
        cursor.execute("""
            SELECT t.id, t.ref_no, t.type, t.transaction_date, l.quantity 
            FROM transactions t
            JOIN purchase_lines l ON t.id = l.transaction_id
            WHERE l.variation_id = %s
            ORDER BY t.transaction_date DESC LIMIT 5
        """, (v_id,))
        for row in cursor.fetchall():
            print(row)

if __name__ == "__main__":
    audit_stock_linkage()
