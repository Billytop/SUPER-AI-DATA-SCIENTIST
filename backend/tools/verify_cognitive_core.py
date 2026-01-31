import http.client
import json

def test_api():
    conn = http.client.HTTPConnection("localhost", 8000)
    headers = {'Content-type': 'application/json'}
    
    query = {
        "query": "Sasa! can you help? My sales are falling and I'm worried about next month.",
        "context": {}
    }
    
    print("Sending Cognitive Query...")
    conn.request("POST", "/ask", json.dumps(query), headers)
    response = conn.getresponse()
    print(f"Status: {response.status}")
    data = response.read().decode()
    print(f"Response Body: {json.dumps(json.loads(data), indent=2)}")
    conn.close()

if __name__ == "__main__":
    test_api()
