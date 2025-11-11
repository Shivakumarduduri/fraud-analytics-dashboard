from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "fraud_detection")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Example dataset — replace this with your actual data or a CSV
data = [
    {"TransactionID": 1, "CustomerID": 101, "Amount": 500, "FraudIndicator": 0},
    {"TransactionID": 2, "CustomerID": 102, "Amount": 12000, "FraudIndicator": 1},
    {"TransactionID": 3, "CustomerID": 103, "Amount": 350, "FraudIndicator": 0},
    {"TransactionID": 4, "CustomerID": 104, "Amount": 2500, "FraudIndicator": 1},
    {"TransactionID": 5, "CustomerID": 105, "Amount": 150, "FraudIndicator": 0}
]

# Insert into MongoDB
collection = db["transactions"]
collection.delete_many({})  # clear old data
collection.insert_many(data)

print("✅ Sample fraud data inserted into MongoDB successfully!")
print("📦 Database:", DB_NAME)
print("🧾 Collection: transactions")
