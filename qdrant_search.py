"""
汎用ID検索スクリプト（削除機能付き）
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

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

PARTIAL_MATCH_FIELDS = {"vector_id", "original_id", "source", "section"}

def match(payload_value, query, partial: bool) -> bool:
    if payload_value is None:
        return False
    pv = str(payload_value)
    if partial:
        return query in pv
    else:
        return pv == query

def main():
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"コレクション: {QDRANT_COLLECTION} ({info.points_count}件)\n")
    
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
    
    use_partial = False
    if field in PARTIAL_MATCH_FIELDS:
        mode = input("一致方式: [1] 部分一致(デフォルト)  [2] 完全一致 → ").strip()
        use_partial = (mode != "2")
    
    match_label = "部分一致" if use_partial else "完全一致"
    print(f"\n検索中... ({match_label})")
    
    all_results = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_results.extend(batch)
        if offset is None:
            break
    
    results = [
        p for p in all_results
        if match(p.payload.get(field), value, use_partial)
    ]
    
    print(f"\n=== 検索結果: {len(results)}件 ===\n")
    
    for i, point in enumerate(results, 1):
        payload = point.payload
        print(f"[{i}] ID: {point.id}")
        print(f"  vector_id:   {payload.get('vector_id', '-')}")
        print(f"  original_id: {payload.get('original_id', '-')}")
        print(f"  type:        {payload.get('type', '-')}")
        print(f"  section:     {payload.get('section', '-')}")
        print(f"  lang:        {payload.get('lang', '-')}")
        title = payload.get('title', '-')
        print(f"  title: {title[:60]}..." if len(str(title)) > 60 else f"  title: {title}")
        print()
    
    if not results:
        return

    # テキスト表示
    show_text = input("テキスト内容を表示しますか？ (y/n): ").strip().lower()
    if show_text == 'y':
        for point in results:
            payload = point.payload
            print("\n" + "=" * 60)
            print(f"【{payload.get('vector_id', '-')}】")
            print("=" * 60)
            text = payload.get('text', payload.get('section_text', '-'))
            print(text[:1000] if len(str(text)) > 1000 else text)
            print()

    # 削除
    print("\n--- 削除オプション ---")
    print("  all  : 検索結果をすべて削除")
    print("  番号  : 指定した番号のみ削除 (例: 1,3,5)")
    print("  n    : 削除しない")
    del_choice = input("選択: ").strip().lower()

    if del_choice == 'n' or del_choice == '':
        print("削除をキャンセルしました。")
        return

    if del_choice == 'all':
        targets = results
    else:
        try:
            indices = [int(x.strip()) - 1 for x in del_choice.split(',')]
            targets = [results[i] for i in indices if 0 <= i < len(results)]
        except ValueError:
            print("無効な入力です。削除をキャンセルしました。")
            return

    if not targets:
        print("対象が見つかりませんでした。")
        return

    # 確認
    print(f"\n以下の {len(targets)} 件を削除します:")
    for p in targets:
        print(f"  - {p.payload.get('vector_id', p.id)}")
    
    confirm = input("\n本当に削除しますか？ (yes/n): ").strip().lower()
    if confirm != 'yes':
        print("削除をキャンセルしました。")
        return

    ids_to_delete = [p.id for p in targets]
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=PointIdsList(points=ids_to_delete)
    )
    print(f"\n✅ {len(ids_to_delete)} 件を削除しました。")

    # 削除後の件数確認
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"   コレクション残件数: {info.points_count}件")

if __name__ == "__main__":
    main()