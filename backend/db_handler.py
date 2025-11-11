from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri)
    db_name = os.getenv("DB_NAME", "fraud_detection")
    return client[db_name]

def save_results_to_db(db, collection_name, df):
    if df.empty:
        return
    records = df.to_dict("records")
    db[collection_name].insert_many(records)
