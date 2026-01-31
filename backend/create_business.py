import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Business
from django.contrib.auth.models import User

admin = User.objects.get(username='admin')

# Create business
business, created = Business.objects.get_or_create(
    name="SephlightyAI Demo Store",
    owner=admin
)

if created:
    print(f"[OK] Created business: {business.name}")
else:
    print(f"[OK] Business already exists: {business.name}")
