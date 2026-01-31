import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.getcwd())

from laravel_modules.ai_brain.central_integration_layer import CentralAI

def verify_bridge():
    brain = CentralAI()
    brain.initialize()
    
    print("\n--- TEST 1: Employee Sales Resolution ---")
    query_1 = "what is the total sales of employee shakira ismail this month"
    result_1 = brain.query(query_1)
    print(f"Query: {query_1}")
    print(f"Response: {result_1.get('response')}")
    print(f"Metadata: {result_1.get('metadata')}")
    
    print("\n--- TEST 2: Report Generation Request ---")
    query_2 = "prepare weekly report and export to pdf"
    result_2 = brain.query(query_2)
    print(f"Query: {query_2}")
    print(f"Response: {result_2.get('response')}")
    
    print("\n--- TEST 3: Report Engine Export ---")
    # Simulate API call to export endpoint logic
    mock_data = [{"Metric": "Sales", "Value": "150,000"}, {"Metric": "Leads", "Value": "45"}]
    pdf_path = brain.reports.generate_master_report(mock_data, "Verification_Report", export_format="pdf")
    print(f"Generated PDF: {pdf_path}")
    
    if os.path.exists(pdf_path):
        print("VERIFICATION SUCCESS: PDF Report found on disk.")
    else:
        print("VERIFICATION FAILURE: PDF Report not found.")

if __name__ == "__main__":
    verify_bridge()
