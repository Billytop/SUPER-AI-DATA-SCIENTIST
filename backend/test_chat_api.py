import requests
import json

url = "http://localhost:8001/api/chat/"
# We need a token. I'll try to find an existing user or just test the 400 path for missing message.

print("--- Testing Chat API (Missing Message) ---")
try:
    res = requests.post(url, json={})
    print(f"Status: {res.status_code}")
    print(f"Response: {res.text}")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Testing Chat API (Unauthenticated) ---")
try:
    res = requests.post(url, json={"message": "hey"})
    print(f"Status: {res.status_code}")
    print(f"Response: {res.text}")
except Exception as e:
    print(f"Error: {e}")
