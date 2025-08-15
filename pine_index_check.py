from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

print("=== List of indexes ===")
indexes = pc.list_indexes()
print(indexes.names())  # すべてのインデックス名がリスト表示される