import requests
import json

BASE_URL = "http://localhost:8000"

def test_cognitive_intelligence():
    test_cases = [
        {
            "name": "Greeting & Help Query",
            "payload": {"query": "Sasa! can you help?", "context": {}}
        },
        {
            "name": "Financial & Temporal Reasoning",
            "payload": {"query": "Habari! How is my profit and revenue looking for next month?", "context": {}}
        },
        {
            "name": "Urgency & Stress Detection",
            "payload": {"query": "URGENT help needed! My inventory is failing and I have many errors!", "context": {}}
        },
        {
            "name": "Logical Engineer-level Inquiry",
            "payload": {"query": "Analyze the dependency mapping between my manufacturing BOMS and crm sales targets.", "context": {}}
        },
        {
            "name": "Business Slang Greeting",
            "payload": {"query": "Niaje SephlightyAI! What is the status today?", "context": {}}
        }
    ]
    
    print("=== STARTING COGNITIVE CORE VERIFICATION ===\n")
    
    for case in test_cases:
        print(f"Testing Case: {case['name']}")
        print(f"Query: {case['payload']['query']}")
        try:
            r = requests.post(f"{BASE_URL}/ask", json=case['payload'])
            if r.status_code == 200:
                data = r.json()
                print(f"Response: {data['response']}")
                print(f"Emotion: {data['metadata']['detected_emotion']}")
                print(f"Reasoning: {data['metadata']['reasoning_applied']}")
            else:
                print(f"Error: Status {r.status_code} - {r.text}")
        except Exception as e:
            print(f"Connection Error: {e}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    test_cognitive_intelligence()
