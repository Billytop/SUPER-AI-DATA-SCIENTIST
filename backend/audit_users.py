from django.db import connections
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def audit_users():
    cursor = connections['erp'].cursor()
    
    print("--- USER AUDIT ---")
    search_terms = ['0014', 'njuku', 'shakira', 'kanda', 'everd', 'moussa']
    
    for term in search_terms:
        print(f"\nSearching for: {term}")
        cursor.execute("""
            SELECT id, username, email, first_name, last_name 
            FROM users 
            WHERE username LIKE %s OR first_name LIKE %s OR last_name LIKE %s OR email LIKE %s
        """, (f"%{term}%", f"%{term}%", f"%{term}%", f"%{term}%"))
        results = cursor.fetchall()
        for row in results:
            print(row)

if __name__ == "__main__":
    audit_users()
