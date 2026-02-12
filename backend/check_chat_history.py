
import os
import django
import sys

# Setup Django Environment
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.contrib.auth.models import User
from chat.models import Conversation, Message
from auth_app.models import UserProfile

def check_history():
    print("--- Checking Chat History ---")
    
    # Get the user (assuming the one from the screenshot/context, or just list all)
    # The user in screenshot is 'BILALI NJUKUNI', email 'njukunibilali@gmail.com'
    # We'll try to find by email or username.
    
    users = User.objects.all()
    print(f"Total Users: {users.count()}")
    
    for user in users:
        print(f"\nUser: {user.username} ({user.email})")
        
        # Check Profile/Org
        try:
            profile = user.profile
            print(f"  - Organization: {profile.organization.name if profile.organization else 'None'}")
        except Exception as e:
            print(f"  - Profile Error: {e}")
            
        # Check Conversations
        conversations = Conversation.objects.filter(user=user)
        print(f"  - Conversations: {conversations.count()}")
        
        for conv in conversations:
            msg_count = conv.messages.count()
            print(f"    - ID: {conv.id} | Title: {conv.title} | Msgs: {msg_count} | Org: {conv.organization.name}")
            
            # Show last few messages
            last_msgs = conv.messages.order_by('-created_at')[:2]
            for m in last_msgs:
                print(f"      - [{m.role}] {m.content[:50]}...")

if __name__ == "__main__":
    check_history()
