import os
from dotenv import load_dotenv
from pinecone import Pinecone

# .envファイルの読み込み
load_dotenv()

# 環境変数からAPIキーを取得
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# インデックス名
INDEX_NAME = "raiden-main"

# Pineconeクライアントのインスタンス化
pc = Pinecone(api_key=PINECONE_API_KEY)

# インデックスへ接続
index = pc.Index(INDEX_NAME)


def clear_index(index):
    print(f"Clearing index: {INDEX_NAME}")

    # 全ベクトル削除
    index.delete(delete_all=True)

    print(f"All vectors in index '{INDEX_NAME}' have been deleted.")


# 実行
if __name__ == "__main__":
    clear_index(index)