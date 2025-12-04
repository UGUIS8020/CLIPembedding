import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# .envファイルの読み込み
load_dotenv()

# 環境変数からAPIキーとURLを取得
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # ローカルの場合
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # クラウド版の場合

# コレクション名
COLLECTION_NAME = "raiden-main"

# Qdrantクライアントのインスタンス化
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY  # ローカル版では不要
)


def clear_collection(client, collection_name):
    print(f"Clearing collection: {collection_name}")

    try:
        # コレクション内の全ポイントを削除
        # 方法1: コレクションごと削除して再作成する場合
        # collection_info = client.get_collection(collection_name)
        # client.delete_collection(collection_name)
        # print(f"Collection '{collection_name}' has been deleted.")
        # 再作成が必要な場合はここで create_collection を実行
        
        # 方法2: 全ポイントのみ削除（コレクション構造は保持）
        from qdrant_client.models import Filter
        
        client.delete(
            collection_name=collection_name,
            points_selector=Filter()  # 空のフィルターで全ポイント選択
        )
        
        print(f"All vectors in collection '{collection_name}' have been deleted.")
        
    except Exception as e:
        print(f"Error: {e}")


# 実行
if __name__ == "__main__":
    clear_collection(client, COLLECTION_NAME)