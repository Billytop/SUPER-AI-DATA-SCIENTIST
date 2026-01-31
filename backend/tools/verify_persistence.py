import http.client
import json
import time
import os

def test_persistence_and_alerts():
    conn = http.client.HTTPConnection("localhost", 8000)
    headers = {'Content-type': 'application/json'}
    
    print("=== OMNIBRAIN PERSISTENCE & ALERT VERIFICATION ===\n")
    
    # 1. Trigger Learning (this should save state)
    print("Step 1: Sending learning feedback...")
    feedback = {"query": "how is the weather", "correction": "irrelevant to business", "category": "OUT_OF_SCOPE"}
    conn.request("POST", "/ask", json.dumps({"query": "Learn this: weather is out of scope.", "context": {"feedback": feedback}}), headers)
    r = conn.getresponse()
    print(f"Learning Status: {r.status}")
    r.read() # drain
    
    # 2. Check if state file exists
    time.sleep(1) # wait for disk write
    state_path = "backend/data/omnibrain_state.json"
    if os.path.exists(state_path):
        print(f"SUCCESS: State file found at {state_path}")
        with open(state_path, "r") as f:
            state = json.load(f)
            print(f"Learned patterns count: {len(state.get('learned_patterns', []))}")
    else:
        print(f"FAILURE: State file NOT found at {state_path}")

    # 3. Verify Dashboard mapping persistence
    print("\nStep 2: Verifying Dashboard KPIs...")
    conn.request("GET", "/saas/dashboard?connection_id=TENANT_001")
    r = conn.getresponse()
    data = json.loads(r.read().decode())
    print(f"Dashboard KPI 1: {data['kpis'][0]['label']} = {data['kpis'][0]['value']}")
    
    conn.close()

if __name__ == "__main__":
    test_persistence_and_alerts()
