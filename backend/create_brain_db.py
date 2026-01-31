
import MySQLdb
import os

def create_db():
    try:
        # Connect to MySQL server (no DB selected)
        db = MySQLdb.connect(
            host=os.environ.get('DB_HOST', '127.0.0.1'),
            user=os.environ.get('DB_USER', 'root'),
            passwd=os.environ.get('DB_PASSWORD', ''),
            port=int(os.environ.get('DB_PORT', 3306))
        )
        cursor = db.cursor()
        
        # Create Database
        cursor.execute("CREATE DATABASE IF NOT EXISTS sephlighty_brain CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print("Database `sephlighty_brain` created (or exists).")
        
    except Exception as e:
        print(f"Error creating DB: {e}")

if __name__ == "__main__":
    create_db()
