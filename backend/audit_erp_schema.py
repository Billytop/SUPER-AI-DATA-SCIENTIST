from django.db import connections
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def audit_erp():
    cursor = connections['erp'].cursor()
    
    print("--- TRANSACTION TABLES ---")
    cursor.execute("SHOW TABLES LIKE '%transaction%'")
    for row in cursor.fetchall():
        print(row[0])
        
    print("\n--- USER TABLES ---")
    cursor.execute("SHOW TABLES LIKE '%user%'")
    for row in cursor.fetchall():
        print(row[0])
        
    print("\n--- SCHEMA OF 'transactions' ---")
    try:
        cursor.execute("DESCRIBE transactions")
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]}")
    except:
        print("Table 'transactions' not found or inaccessible.")

if __name__ == "__main__":
    audit_erp()
