try:
    import MySQLdb
except ImportError:
    import pymysql
    pymysql.install_as_MySQLdb()
    import MySQLdb
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.environ.get('DATABASE_HOST', 'localhost')
USER = os.environ.get('DATABASE_USER', 'root')
PASSWORD = os.environ.get('DATABASE_PASSWORD', '')
PORT = int(os.environ.get('DATABASE_PORT', 3306))
# Updated DB Name
DB_NAME = 'sephlighty_2026'

try:
    conn = MySQLdb.connect(host=HOST, user=USER, passwd=PASSWORD, port=PORT)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    print(f"Database '{DB_NAME}' checked/created successfully.")
    conn.close()
except Exception as e:
    print(f"Error connecting to MySQL: {e}")
    exit(1)
