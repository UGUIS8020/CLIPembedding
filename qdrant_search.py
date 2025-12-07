"""
汎用ID検索スクリプト
vector_id, original_id, pmid など任意のフィールドで検索可能
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_COLLECTION = "raiden-main"

FIELDS = [
    "vector_id",
    "original_id", 
    "pmid",
    "type",
    "source",
    "section",
    "lang"
]

def main():
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # コレクション情報
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"コレクション: {QDRANT_COLLECTION} ({info.points_count}件)\n")
    
    # フィールド選択
    print("検索フィールド:")
    for i, field in enumerate(FIELDS, 1):
        print(f"  {i}. {field}")
    
    choice = input("\n番号を選択: ").strip()
    
    try:
        field = FIELDS[int(choice) - 1]
    except (ValueError, IndexError):
        print("無効な選択です")
        return
    
    value = input(f"{field} の値: ").strip()
    
    # 全データ取得してフィルタリング
    print("\n検索中...")
    all_results, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=10000
    )
    
    results = [p for p in all_results if p.payload.get(field) == value]
    
    print(f"\n=== 検索結果: {len(results)}件 ===\n")
    
    for point in results:
        payload = point.payload
        print(f"ID: {point.id}")
        print(f"  vector_id: {payload.get('vector_id', '-')}")
        print(f"  original_id: {payload.get('original_id', '-')}")
        print(f"  type: {payload.get('type', '-')}")
        print(f"  section: {payload.get('section', '-')}")
        print(f"  lang: {payload.get('lang', '-')}")
        title = payload.get('title', '-')
        print(f"  title: {title[:50]}..." if len(title) > 50 else f"  title: {title}")
        print()
    
    # テキスト内容を表示するか確認
    if results:
        show_text = input("テキスト内容を表示しますか？ (y/n): ").strip().lower()
        if show_text == 'y':
            for point in results:
                payload = point.payload
                print("\n" + "=" * 60)
                print(f"【{payload.get('vector_id', '-')}】")
                print("=" * 60)
                text = payload.get('text', payload.get('section_text', '-'))
                print(text[:1000] if len(text) > 1000 else text)
                print()

if __name__ == "__main__":
    main()