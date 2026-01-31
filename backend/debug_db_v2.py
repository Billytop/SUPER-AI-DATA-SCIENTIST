
import os
import django
from django.db import connection

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def check_db():
    print(f"DB Name in settings: {connection.settings_dict['NAME']}")
    print(f"DB Host in settings: {connection.settings_dict['HOST']}")
    
    with connection.cursor() as cursor:
        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()[0]
        print(f"Connected to MySQL Database: {db_name}")
        
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Total tables: {len(tables)}")
        
        if 'auth_user' in tables:
            print("[OK] auth_user table exists.")
            from django.contrib.auth.models import User
            print(f"User count: {User.objects.count()}")
            if User.objects.filter(username='admin').exists():
                 print("[OK] Admin user found.")
            else:
                 print("[FAIL] Admin user NOT found.")
        else:
            print("[CRITICAL] auth_user table MISSING!")
            print("Tables found:", tables)

if __name__ == '__main__':
    check_db()
