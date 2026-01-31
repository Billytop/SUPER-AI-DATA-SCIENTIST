
import os
import django
from django.db import connection, connections

# Ensure setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

CANDIDATES = [
    "2026", "2026v4", "sephlighty_ai", "sephylightlive", "livedb", 
    "sephlighty 2026", "sephlighty_2026", "new2026", "hq", "sephylight"
]

def check_all_dbs():
    print("--- Scanning Databases for Sales Data ---")
    
    # We need to manually cycle the connection settings or use raw SQL to switch DBs if user permissions allow
    # Since we are likely root (based on .env), we can usually just USE <db>;
    
    with connection.cursor() as cursor:
        for db in CANDIDATES:
            try:
                # Switch DB
                cursor.execute(f"USE `{db}`")
                
                # Check table existence first to avoid error
                cursor.execute("SHOW TABLES LIKE 'sales_transaction'")
                if cursor.fetchone():
                    cursor.execute("SELECT COUNT(*) FROM sales_transaction")
                    count = cursor.fetchone()[0]
                    print(f"[{db}]: {count} rows")
                    
                    if count > 0:
                        cursor.execute("SELECT MAX(transaction_date) FROM sales_transaction")
                        last_date = cursor.fetchone()[0]
                        print(f"   Last Transaction: {last_date}")
                else:
                    # Try legacy name 'transactions' just in case
                    cursor.execute("SHOW TABLES LIKE 'transactions'")
                    if cursor.fetchone():
                         cursor.execute("SELECT COUNT(*) FROM transactions")
                         count = cursor.fetchone()[0]
                         print(f"[{db}]: {count} rows (Legacy Table 'transactions')")
                    else:
                        print(f"[{db}]: Table not found")
                        
            except Exception as e:
                print(f"[{db}]: Error - {e}")

if __name__ == "__main__":
    check_all_dbs()
