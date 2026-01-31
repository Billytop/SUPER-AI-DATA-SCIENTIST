
import os
import django
from django.db import connection

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

def check():
    print(f"Checking DB: {os.environ.get('DB_NAME')}")
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DATABASE()")
            print(f"Connected to: {cursor.fetchone()[0]}")
            
            cursor.execute("SELECT COUNT(*) FROM transactions")
            print(f"Transactions count: {cursor.fetchone()[0]}")
            
            cursor.execute("SELECT SUM(final_total) FROM transactions WHERE type='sell'")
            print(f"Total Sales: {cursor.fetchone()[0]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check()
