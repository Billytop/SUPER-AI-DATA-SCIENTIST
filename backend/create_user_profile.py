
import os
import django

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth import get_user_model
from auth_app.models import Organization, UserProfile

def setup_profile():
    User = get_user_model()
    email = "admin@sephlighty.ai"
    
    try:
        user = User.objects.get(email=email)
        print(f"Found user: {user.email}")
        
        # 1. Create Organization
        org, created = Organization.objects.get_or_create(
            slug="sephlighty-hq",
            defaults={
                "name": "Sephlighty HQ",
                "subscription_tier": "enterprise"
            }
        )
        if created:
            print("Created Organization: Sephlighty HQ")
        else:
            print("Found Organization: Sephlighty HQ")
            
        # 2. Create Profile
        profile, p_created = UserProfile.objects.get_or_create(
            user=user,
            defaults={
                "organization": org,
                "role": "admin"
            }
        )
        
        if p_created:
            print("Created UserProfile linked to Organization.")
        else:
            print("UserProfile already exists.")
            
    except User.DoesNotExist:
        print(f"User {email} not found! Run create_admin.py first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    setup_profile()
