
import os
import django
from django.contrib.auth import get_user_model

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

User = get_user_model()
try:
    u = User.objects.get(username='admin')
    u.set_password('admin123')
    u.save()
    print(f"Password for user '{u.username}' has been reset to 'admin123'.")
except User.DoesNotExist:
    print("User 'admin' does not exist.")
