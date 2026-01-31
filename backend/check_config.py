
import os
import django
from django.conf import settings

# Force Django to read the actual settings file, not just cache
import sys
if "DJANGO_SETTINGS_MODULE" not in os.environ:
    os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"

try:
    django.setup()
    print(f"Configured DB_NAME: {settings.DATABASES['default']['NAME']}")
    
    from django.db import connection
    with connection.cursor() as cursor:
        cursor.execute("SELECT DATABASE()")
        print(f"Actual Active DB: {cursor.fetchone()[0]}")
except Exception as e:
    print(f"Error: {e}")
