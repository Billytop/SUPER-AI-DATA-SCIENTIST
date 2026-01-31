
import requests
import json

url = "http://localhost:8001/api/chat/"
payload = {"message": "total sales this year"}
headers = {
    "Content-Type": "application/json",
    # Assuming no auth needed for this specific test or need to login first
    # If auth is needed, we might need to grab a token first, but let's try without first or mocking
}

# Login to get token first
login_url = "http://localhost:8001/api/auth/login/"
login_payload = {"email": "admin@sephlighty.ai", "password": "password123"} 

try:
    print("Logging in...")
    # Adjust login payload according to your actual login endpoint expectations
    session = requests.Session()
    # Attempting to login manually or just use a known token? 
    # Let's try to just hit the endpoint, if it fails 401/403 we know server is up at least.
    # To properly test content, we need a valid token.
    
    # We will try to bypass auth for a second in the view if testing, OR assuming the previous runserver has the DB with this user.
    # Let's assume we need a token.
    
    resp = requests.post(login_url, json=login_payload)
    if resp.status_code == 200:
        token = resp.json().get('access')
        headers['Authorization'] = f"Bearer {token}"
        print("Login successful.")
    else:
        print(f"Login failed: {resp.status_code} {resp.text}")
        # Proceeding might fail but let's see
    
    print("Sending chat message...")
    response = requests.post(url, json=payload, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))

except Exception as e:
    print(f"Error: {e}")
