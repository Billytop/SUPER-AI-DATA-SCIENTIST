import os
import django
from django.conf import settings
import pandas as pd

# Configure simple settings
if not settings.configured:
    settings.configure(DEBUG=True)

try:
    import openpyxl
    print("OpenPyXL version:", openpyxl.__version__)
except ImportError:
    print("OpenPyXL NOT installed!")

try:
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    print("Pandas created.")
    
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    output_path = os.path.join(downloads, "test_export.xlsx")
    
    print(f"Attempting export to {output_path}...")
    df.to_excel(output_path, index=False, engine='openpyxl')
    print("Success! File created.")
    
except Exception as e:
    print(f"Export FAILED: {e}")
