from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
import os

brain = OmnibrainSaaSEngine()
tables = brain._execute_erp_query("SHOW TABLES")
print("TABLES FOUND:")
for row in tables:
    print(list(row.values())[0])
