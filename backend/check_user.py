import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.contrib.auth.models import User
from auth_app.models import UserProfile

def check_user(search):
    u = User.objects.filter(username=search).first() or User.objects.filter(email=search).first()
    if u:
        print(f"USERFOUND: {u.username}")
        print(f"EMAIL: {u.email}")
        has_p = hasattr(u, "profile")
        print(f"HAS_PROFILE: {has_p}")
        if has_p:
            print(f"ORG: {u.profile.organization}")
    else:
        print(f"USER NOT FOUND: {search}")

if __name__ == "__main__":
    check_user('njukunibilali')
    check_user('njukunibilali@gmail.com')
