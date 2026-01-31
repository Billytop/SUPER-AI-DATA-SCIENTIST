
import os
import django
from django.contrib.auth import get_user_model

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

User = get_user_model()
try:
    u = User.objects.get(username='admin')
    print(f"Username: {u.username}")
    print(f"Email: {u.email}")
    print(f"Is Active: {u.is_active}")
    print(f"Has Usable Password: {u.has_usable_password()}")
    
    # Try authenticating
    from django.contrib.auth import authenticate
    auth_user = authenticate(username='admin', password='admin123')
    print(f"Authentication (admin/admin123): {'SUCCESS' if auth_user else 'FAILED'}")
    
except User.DoesNotExist:
    print("User 'admin' does not exist.")
