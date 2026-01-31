from django.db import connections
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def audit_inventory():
    cursor = connections['erp'].cursor()
    
    print("--- INVENTORY/STOCK TABLES ---")
    cursor.execute("SHOW TABLES LIKE '%stock%'")
    for row in cursor.fetchall():
        print(row[0])
        
    cursor.execute("SHOW TABLES LIKE '%product%'")
    for row in cursor.fetchall():
        print(row[0])

    cursor.execute("SHOW TABLES LIKE '%variation%'")
    for row in cursor.fetchall():
        print(row[0])
        
    print("\n--- SCHEMA OF 'products' (or similar) ---")
    try:
        cursor.execute("DESCRIBE products")
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]}")
    except:
        print("Table 'products' not found.")

    print("\n--- SEARCHING FOR 'vin tape' ---")
    try:
        cursor.execute("SELECT id, name, sku FROM products WHERE name LIKE '%vin tape%'")
        for row in cursor.fetchall():
            print(row)
    except:
        pass

if __name__ == "__main__":
    audit_inventory()
