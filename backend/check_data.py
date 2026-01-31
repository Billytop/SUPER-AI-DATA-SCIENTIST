
import os
import django
from django.db import connection

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

def check_data():
    with connection.cursor() as cursor:
        print("--- Checking Sales Transaction Data ---")
        
        # 1. Total count
        cursor.execute("SELECT COUNT(*) FROM sales_transaction")
        count = cursor.fetchone()[0]
        print(f"Total rows in sales_transaction: {count}")
        
        if count > 0:
            # 2. Check date range
            cursor.execute("SELECT MIN(transaction_date), MAX(transaction_date) FROM sales_transaction")
            min_date, max_date = cursor.fetchone()
            print(f"Date Range: {min_date} to {max_date}")
            
            # 3. Check 2026 data
            cursor.execute("SELECT COUNT(*), SUM(final_total) FROM sales_transaction WHERE YEAR(transaction_date) = 2026 AND type='sell'")
            cnt_2026, sum_2026 = cursor.fetchone()
            print(f"2026 Sales Count: {cnt_2026}")
            print(f"2026 Sales Sum: {sum_2026}")
            
            # 4. Sample row
            cursor.execute("SELECT id, type, final_total, transaction_date FROM sales_transaction LIMIT 5")
            print("\nSample Rows:")
            for row in cursor.fetchall():
                print(row)

if __name__ == "__main__":
    check_data()
