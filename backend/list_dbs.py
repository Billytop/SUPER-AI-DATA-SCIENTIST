
import os
import django
from django.db import connection

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

def list_dbs():
    with connection.cursor() as cursor:
        cursor.execute("SHOW DATABASES")
        dbs = cursor.fetchall()
        print("Available Databases:")
        for db in dbs:
            print(f"- {db[0]}")

if __name__ == "__main__":
    list_dbs()
