
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.contrib.auth.models import User
from auth_app.models import Organization, UserProfile

def fix_admin_profile():
    try:
        user = User.objects.get(username='admin')
        print(f"Found user: {user.username}")
        
        # Check if profile exists
        if hasattr(user, 'profile'):
            print("User already has a profile.")
            return

        print("User has no profile. Creating one...")
        
        # Get or create organization
        org, created = Organization.objects.get_or_create(
            name="Sephlighty HQ",
            defaults={
                'slug': 'sephlighty-hq',
                'subscription_tier': 'enterprise'
            }
        )
        if created:
            print(f"Created organization: {org.name}")
        else:
            print(f"Using existing organization: {org.name}")
            
        # Create profile
        profile = UserProfile.objects.create(
            user=user,
            organization=org,
            role='admin'
        )
        print(f"Created profile for {user.username}")
        
    except User.DoesNotExist:
        print("User 'admin' does not exist. Please run create_admin.py first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    fix_admin_profile()
