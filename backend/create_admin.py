
import os
import django

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth import get_user_model

def create_admin():
    User = get_user_model()
    email = "admin@sephlighty.ai"
    password = "password123"
    
    if not User.objects.filter(email=email).exists():
        print(f"Creating user: {email}")
        try:
            # Create superuser
            User.objects.create_superuser(
                username="admin",
                email=email,
                password=password
            )
            print(f"User created successfully.\nEmail: {email}\nPassword: {password}")
        except Exception as e:
            print(f"Error creating user: {e}")
    else:
        print(f"User {email} already exists.")
        # Optional: Reset password if known user exists but login fails
        u = User.objects.get(email=email)
        u.set_password(password)
        u.save()
        print(f"Password reset to: {password}")

if __name__ == "__main__":
    create_admin()
