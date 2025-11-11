from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print("✅ Connected to MongoDB Atlas successfully!")
    print("📦 Database name:", db.name)
    print("📚 Collections available:", db.list_collection_names())
except Exception as e:
    print("❌ Connection failed:", e)
