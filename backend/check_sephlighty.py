
import os
import django
from django.db import connection

# Force override DB_NAME for this check
os.environ["DB_NAME"] = "sephlighty_ai"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

def check_sephlighty_ai():
    print(f"Checking database: {os.environ['DB_NAME']}")
    try:
        with connection.cursor() as cursor:
            # 1. Total count
            cursor.execute("SELECT COUNT(*) FROM sales_transaction")
            count = cursor.fetchone()[0]
            print(f"Total rows in sales_transaction: {count}")
            
            if count > 0:
                cursor.execute("SELECT MIN(transaction_date), MAX(transaction_date) FROM sales_transaction")
                min_date, max_date = cursor.fetchone()
                print(f"Date Range: {min_date} to {max_date}")
    except Exception as e:
        print(f"Error accessing database: {e}")

if __name__ == "__main__":
    check_sephlighty_ai()
