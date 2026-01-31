
import os
import django
import sys

# Setup Django environment
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

try:
    print("Attempting to import SQLReasoningAgent...")
    from reasoning.agents import SQLReasoningAgent
    print("Import successful.")
    
    # Try basic instantiation if possible (might need args)
    # Checking signature... usually user_id
    # agent = SQLReasoningAgent(user_id=1) 
    # print("Instantiation successful.")
    
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
