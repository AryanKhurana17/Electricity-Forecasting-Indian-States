from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get MongoDB connection URL from .env
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

def test_mongodb_connection():
    try:
        client = MongoClient(MONGO_DB_URL)
        # Ping the server
        client.admin.command("ping")
        print("MongoDB connection successful!")
    except Exception as e:
        print("MongoDB connection failed:", e)

if __name__ == "__main__":
    test_mongodb_connection()