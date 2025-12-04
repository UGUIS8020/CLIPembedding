import os
import time
import uuid
from dotenv import load_dotenv

from pinecone import Pinecone
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

# =========================
# 設定
# =========================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "raiden-main"
PINECONE_NAMESPACE = ""  # デフォルト namespace

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "raiden-main"

VECTOR_DIM = 1536
QDRANT_DISTANCE = Distance.COSINE

# Qdrant に一度に投げるポイント数
MAX_QDRANT_BATCH = 64
MAX_RETRIES = 5

# デバッグモード
DEBUG_MODE = False


def validate_env():
    missing = []
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise RuntimeError(f"環境変数が不足しています: {', '.join(missing)}")

    print(f"QDRANT_URL = {QDRANT_URL}")
    print(f"QDRANT_API_KEY set? = {bool(QDRANT_API_KEY)}")


def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_host = os.getenv("PINECONE_INDEX_HOST")
    if index_host:
        index = pc.Index(host=index_host)
        print(f"Using Pinecone index host: {index_host}")
    else:
        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"Using Pinecone index name: {PINECONE_INDEX_NAME}")
    return index


def init_qdrant():
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=180  # タイムアウトを180秒に設定
    )
    
    # 接続テスト
    try:
        collections = client.get_collections()
        print(f"✅ Qdrant接続成功。コレクション数: {len(collections.collections)}")
    except Exception as e:
        print(f"❌ Qdrant接続エラー: {e}")
        raise

    # すでにコレクションがあれば「そのまま使う」
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=QDRANT_DISTANCE,
            ),
        )
        print(f"Qdrant collection '{QDRANT_COLLECTION}' を新規作成しました")
    else:
        info = client.get_collection(QDRANT_COLLECTION)
        count = getattr(info, "points_count", None)
        if count is None:
            count = getattr(info, "vectors_count", None)
        print(
            f"Qdrant collection '{QDRANT_COLLECTION}' は既に存在します "
            f"(現在のベクトル数: {count if count is not None else 'unknown'})"
        )
        print("既存データは削除せず、上書きしながら追加します。")

    return client


def to_uuid_from_pinecone_id(vid: str) -> str:
    """Pinecone の string ID を Qdrant 用の UUID に変換（決定的）"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"raiden-main:{vid}"))


def flatten_payload(metadata: dict, original_id: str) -> dict:
    """
    ネスト構造のpayloadをフラット化する
    
    Before:
    {
        "page_content": "...",
        "metadata": {
            "type": "content",
            "weight": 1.0,
            "title": "...",
            ...
        }
    }
    
    After:
    {
        "text": "...",
        "type": "content",
        "weight": 1.0,
        "title": "...",
        "original_id": "...",
        ...
    }
    """
    # デバッグ: 最初の数件で構造を確認
    if not hasattr(flatten_payload, 'debug_count'):
        flatten_payload.debug_count = 0
    
    if flatten_payload.debug_count < 3:
        print(f"\n=== Payload 構造デバッグ {flatten_payload.debug_count + 1} ===")
        print(f"Keys: {list(metadata.keys())}")
        flatten_payload.debug_count += 1
    
    flattened = {}
    
    # ネスト構造の場合
    if "metadata" in metadata:
        # page_content を text に変換
        if "page_content" in metadata:
            flattened["text"] = metadata["page_content"]
        
        # metadata の中身を全て展開
        nested_metadata = metadata["metadata"]
        if isinstance(nested_metadata, dict):
            flattened.update(nested_metadata)
        
        # metadata以外のトップレベルフィールドも保持
        for key, value in metadata.items():
            if key not in ["metadata", "page_content"]:
                flattened[key] = value
    
    # 既にフラットな構造の場合
    else:
        flattened = metadata.copy()
        
        # page_content が存在したら text に変換
        if "page_content" in flattened:
            flattened["text"] = flattened.pop("page_content")
    
    # original_id を必ず追加
    flattened["original_id"] = original_id
    
    return flattened


def safe_upsert(qdrant: QdrantClient, points_batch):
    """502やタイムアウトが出たらリトライしながら upsert する"""
    batch_size = len(points_batch)
    
    if DEBUG_MODE:
        print(f"  📤 バッチアップロード開始: {batch_size}件")
        # ペイロードサイズの概算
        total_payload_size = sum(len(str(p.payload)) for p in points_batch)
        print(f"  📊 推定ペイロードサイズ: {total_payload_size:,} バイト")
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start_time = time.time()
            qdrant.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points_batch,
                wait=True,
            )
            elapsed = time.time() - start_time
            if DEBUG_MODE:
                print(f"  ✅ アップロード成功: {elapsed:.2f}秒")
            return
        except UnexpectedResponse as e:
            if getattr(e, "status_code", None) == 502 and attempt < MAX_RETRIES:
                sleep_sec = 2 * attempt
                print(
                    f"  ⚠️ 502 Bad Gateway "
                    f"(attempt {attempt}/{MAX_RETRIES}) -> {sleep_sec}秒待機"
                )
                time.sleep(sleep_sec)
                continue
            raise
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["timeout", "timed out", "connection"]) and attempt < MAX_RETRIES:
                sleep_sec = 5 * attempt  # より長い待機時間
                print(
                    f"  ⚠️ 接続/タイムアウトエラー "
                    f"(attempt {attempt}/{MAX_RETRIES}) -> {sleep_sec}秒待機"
                )
                print(f"     エラー詳細: {str(e)[:100]}...")
                time.sleep(sleep_sec)
                continue
            print(f"❌ upsert 中に回復不能エラー: {e}")
            raise


def migrate():
    validate_env()

    index = init_pinecone()
    qdrant = init_qdrant()

    stats = index.describe_index_stats()
    total_vectors = stats.get("total_vector_count")
    metric = stats.get("metric", "unknown")
    dim = stats.get("dimension")
    print(f"Pinecone index stats: total={total_vectors}, dim={dim}, metric={metric}")

    if dim != VECTOR_DIM:
        print(f"警告: Pinecone の次元数 {dim} と Qdrant の設定 {VECTOR_DIM} が一致していません。")

    print("\n🔄 Pinecone から Qdrant への移行を開始します...")
    print("📝 Payload構造をフラット化しながら移行します\n")

    migrated_count = 0
    batch_no = 0

    try:
        id_generator = index.list(namespace=PINECONE_NAMESPACE)
    except Exception as e:
        print("index.list() に失敗しました。serverless ではないか、古いクライアントの可能性があります。")
        print("その場合は、別途 ID リストをどこかに保存しているか確認してください。")
        raise

    for id_batch in id_generator:
        batch_no += 1
        if not id_batch:
            continue

        fetch_res = index.fetch(ids=id_batch, namespace=PINECONE_NAMESPACE)

        if isinstance(fetch_res, dict):
            vectors_dict = fetch_res.get("vectors", {})
        else:
            vectors_dict = getattr(fetch_res, "vectors", {})

        points = []
        for vid, record in vectors_dict.items():
            if isinstance(record, dict):
                values = record.get("values", [])
                metadata = record.get("metadata", {})
            else:
                values = getattr(record, "values", [])
                metadata = getattr(record, "metadata", {})

            # メタデータをそのまま使用（フラット化は不要）
            # original_id だけ追加
            flattened_payload = metadata.copy()
            flattened_payload["original_id"] = vid

            qdrant_id = to_uuid_from_pinecone_id(vid)

            points.append(
                PointStruct(
                    id=qdrant_id,
                    vector=values,
                    payload=flattened_payload,
                )
            )

        if not points:
            continue

        # Qdrant 用にさらに細かいバッチに分割して upsert
        print(f"\n📦 Batch {batch_no}: {len(points)} 件を {MAX_QDRANT_BATCH} 件ずつ分割してアップロード")
        
        for i in range(0, len(points), MAX_QDRANT_BATCH):
            sub_points = points[i:i + MAX_QDRANT_BATCH]
            sub_batch_no = (i // MAX_QDRANT_BATCH) + 1
            total_sub_batches = (len(points) + MAX_QDRANT_BATCH - 1) // MAX_QDRANT_BATCH
            
            print(f"  📤 サブバッチ {sub_batch_no}/{total_sub_batches}")
            safe_upsert(qdrant, sub_points)
            migrated_count += len(sub_points)
            
            # 各バッチ間に待機時間を追加（サーバー負荷軽減）
            time.sleep(1.0)

        print(f"✅ Batch {batch_no} 完了 (累計 {migrated_count}/{total_vectors})")

    print(f"\n✅ 移行完了: 合計 {migrated_count} ベクトルを Qdrant にコピーしました。")
    print("\n📊 移行後のデータ構造を確認してください:")
    print("python check_vector_ids.py")


if __name__ == "__main__":
    migrate()