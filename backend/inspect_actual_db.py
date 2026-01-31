import os
import django
from django.db import connection

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def list_tables_and_columns():
    try:
        with connection.cursor() as cursor:
            # List all tables
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"Connected to Database: {connection.settings_dict['NAME']}")
            print(f"Found {len(tables)} tables:")
            
            for table in tables:
                table_name = table[0]
                print(f"- {table_name}")
                
            # Inspect specific likely sales tables
            for table in tables:
                t = table[0].lower()
                if t in ['variations']:
                    print(f"\nScanning table: {t}")
                    cursor.execute(f"DESCRIBE `{t}`")
                    cols = cursor.fetchall()
                    col_names = [c[0] for c in cols]
                    print(f"  Columns: {', '.join(col_names)}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_tables_and_columns()
