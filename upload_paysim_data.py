import pandas as pd
from pymongo import MongoClient

# ✅ MongoDB Atlas connection string
client = MongoClient("mongodb+srv://frauduser:Shivakumaryadav@fraudulantappccuster.vct3mvc.mongodb.net/?appName=fraudulantappcCuster")

# ✅ Database name
db = client["fraud_detection"]

# ✅ Path to your dataset (update if needed)
csv_path = r"C:\Users\shiva\fraudulant_app\data\PS_20174392719_1491204439457_log.csv"

# ✅ Load CSV into DataFrame
print("📥 Loading dataset...")
data = pd.read_csv(csv_path)
print("✅ Dataset loaded successfully! Shape:", data.shape)

# ✅ Convert to dictionary for MongoDB
records = data.to_dict(orient="records")

# ✅ Upload to MongoDB
print("🚀 Uploading data to MongoDB Atlas (this may take some time)...")
db["paysim_data"].insert_many(records)

print(f"✅ Successfully inserted {len(records)} records into MongoDB Atlas collection 'paysim_data'!")
