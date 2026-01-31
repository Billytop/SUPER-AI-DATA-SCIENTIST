import http.client
import json

def test_omnibrain_saas():
    conn = http.client.HTTPConnection("localhost", 8000)
    headers = {'Content-type': 'application/json'}
    
    test_cases = [
        {
            "name": "SaaS Dashboard Inquiry",
            "endpoint": "/saas/dashboard",
            "method": "GET",
            "query": ""
        },
        {
            "name": "System Stress Test",
            "endpoint": "/saas/stress-test",
            "method": "POST",
            "query": ""
        },
        {
            "name": "AI Financial Audit",
            "endpoint": "/saas/audit",
            "method": "POST",
            "query": {"query": "Audit my Q4 sales vs inventory"}
        },
        {
            "name": "Confidence Score Verification (Low Confidence)",
            "endpoint": "/ask",
            "method": "POST",
            "query": {"query": "xyz123", "context": {"connection_id": "TENANT_001"}}
        },
        {
            "name": "High Context Cognitive + SaaS Response",
            "endpoint": "/ask",
            "method": "POST",
            "query": {"query": "Sasa! can you predict next month sales?", "context": {"connection_id": "TENANT_001"}}
        }
    ]
    
    print("=== OMNIBRAIN SAAS CORE VERIFICATION ===\n")
    
    for case in test_cases:
        print(f"Testing: {case['name']}")
        if case['method'] == "GET":
            conn.request(case['method'], case['endpoint'])
        else:
            conn.request(case['method'], case['endpoint'], json.dumps(case['query']), headers)
            
        r = conn.getresponse()
        data = json.loads(r.read().decode())
        print(f"Status: {r.status}")
        print(f"Payload: {json.dumps(data, indent=2)}")
        print("-" * 50 + "\n")
    
    conn.close()

if __name__ == "__main__":
    test_omnibrain_saas()
