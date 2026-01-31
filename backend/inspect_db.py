import os
import django
from django.db import connection

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def inspect():
    with connection.cursor() as cursor:
        print("CHECKING 'Majenga' in 'sephlighty 2026' -> 'contacts'...")
        # Note: default db is sephlighty 2026
        
        # Check Columns first to see if 'name' exists
        cursor.execute("DESCRIBE contacts")
        cols = [c[0] for c in cursor.fetchall()]
        print(f"Columns: {cols}")
        
        # Build query based on columns
        # UltimatePOS usually has 'name' (business) or 'first_name'/'last_name'
        where_clauses = []
        if 'name' in cols: where_clauses.append("name LIKE '%Majenga%'")
        if 'first_name' in cols: where_clauses.append("first_name LIKE '%Majenga%'")
        if 'last_name' in cols: where_clauses.append("last_name LIKE '%Majenga%'")
        
        sql = f"SELECT * FROM contacts WHERE {' OR '.join(where_clauses)}"
        print(f"Query: {sql}")
        
        cursor.execute(sql)
        rows = cursor.fetchall()
        if rows:
            print("✅ FOUND MAJENGA!")
            for r in rows:
                print(r)
        else:
            print("❌ Majenga NOT found in this DB.")

if __name__ == "__main__":
    inspect()
