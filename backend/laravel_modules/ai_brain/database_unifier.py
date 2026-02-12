
import os
import mysql.connector

# SOVEREIGN DATABASE UNIFIER v1.0
# The Bridge between AI Logic and Real World SQL Data.

class DatabaseUnifier:
    def __init__(self):
        self.db_config = {
            "host": os.environ.get('DB_HOST', '127.0.0.1'),
            "user": os.environ.get('DB_USER', 'root'),
            "password": os.environ.get('DB_PASSWORD', ''),
            "database": os.environ.get('DB_NAME_ERP', '2026v4')
        }
        self.connection = None

    def connect(self) -> bool:
        """Establishes connection to the real ERP DB."""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            return True
        except Exception:
            return False

    def query_real_data(self, table: str, condition: str = None) -> list:
        """
        Executes a safe SELECT query on the real database.
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return [{"error": "DB Connection Failed"}]
                
        cursor = self.connection.cursor(dictionary=True)
        query = f"SELECT * FROM {table} LIMIT 5"
        if condition:
            # Naive injection protection for demo
            safe_cond = condition.replace(";", "").replace("--", "")
            query = f"SELECT * FROM {table} WHERE {safe_cond} LIMIT 5"
            
        try:
            cursor.execute(query)
            return cursor.fetchall()
        except Exception as e:
            return [{"error": str(e)}]
        finally:
            cursor.close()

    def unified_search(self, term: str) -> str:
        """
        Searches Real DB first, then Sovereign Knowledge.
        """
        # 1. Try Real DB (Products)
        real_results = self.query_real_data("products", f"name LIKE '%{term}%'")
        if real_results and "error" not in real_results[0]:
            return f"### [REAL DATABASE RESULT]\nFound {len(real_results)} matches in ERP:\n{real_results[0]}"
            
        # 2. Try Real DB (Users)
        user_results = self.query_real_data("users", f"name LIKE '%{term}%'")
        if user_results and "error" not in user_results[0]:
            return f"### [REAL DATABASE RESULT]\nFound User match:\n{user_results[0]}"

        return "No direct match in Real DB. Switching to Sovereign Knowledge..."

DB_BRIDGE = DatabaseUnifier()
