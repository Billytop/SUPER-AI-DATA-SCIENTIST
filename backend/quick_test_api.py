import requests
import json

def test():
    urls = [
        "http://localhost:8000/",
        "http://localhost:8000/modules",
        "http://localhost:8000/crm/predict_churn?customer_id=123"
    ]
    
    for url in urls:
        print(f"Testing {url}...")
        try:
            r = requests.get(url) if "crm" not in url else requests.post(url)
            print(f"Status: {r.status_code}")
            print(f"Response: {json.dumps(r.json(), indent=2)}")
        except Exception as e:
            print(f"Error testing {url}: {e}")
        print("-" * 20)

if __name__ == "__main__":
    test()
