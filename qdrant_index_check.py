from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

# 環境変数からURLとAPIキーを取得
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Qdrantクライアントのインスタンス化
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("=== List of collections ===")
collections = client.get_collections()
collection_names = [collection.name for collection in collections.collections]
print(collection_names)

# 詳細情報を表示する場合
print("\n=== Detailed collection info ===")
for collection_name in collection_names:
    collection_info = client.get_collection(collection_name)
    print(f"Name: {collection_name}")
    print(f"Points count: {collection_info.points_count}")
    print(f"Status: {collection_info.status}")
    print("---")