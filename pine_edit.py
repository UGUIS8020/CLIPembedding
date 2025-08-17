#pineconeに保存されたベクトルデータimage,figure_descriptionのweightを0.7に更新するスクリプトです。

import os
from dotenv import load_dotenv
from pinecone import Pinecone
import argparse
from math import ceil

PINECONE_INDEX ="raiden-main" 

def display_page(vector_ids, page_size, page_number):
    total_pages = (len(vector_ids) + page_size - 1) // page_size
    start_idx = (page_number - 1) * page_size
    end_idx = min(start_idx + page_size, len(vector_ids))
    
    print(f"\nページ {page_number}/{total_pages}")
    print(f"表示範囲: {start_idx + 1}～{end_idx} / 全{len(vector_ids)}件")
    print("\nベクトルID一覧:")
    
    for i, vector_id in enumerate(vector_ids[start_idx:end_idx], start=start_idx + 1):
        print(f"{i}. {vector_id}")
    
    print(f"\n--- ページ {page_number}/{total_pages} ---")
    return total_pages

def get_vector_ids_and_update_weight(page_size=50, page_number=1, new_weight=0.7):
    load_dotenv()
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(PINECONE_INDEX)

    stats = index.describe_index_stats()
    total_vector_count = stats.total_vector_count
    dimension = stats.dimension
    
    print(f"インデックスの統計情報:")
    print(f"総ベクトル数: {total_vector_count}")
    print(f"ベクトルの次元数: {dimension}")
    current_size_mb = (total_vector_count * dimension * 4) / (1024*1024)
    max_size_mb = 2048
    remaining_ratio = max_size_mb / current_size_mb
    
    print(f"推定使用容量: {current_size_mb:.2f} MB")
    print(f"最大容量: {max_size_mb} MB (2GB)")
    print(f"現在の使用量で約{remaining_ratio:.1f}倍のデータを保存可能\n")
    
    vector_ids = []
    try:
        list_result = index.list(prefix="", limit=total_vector_count)
        vector_ids = list(list_result.vectors.keys())
        print(f"取得したベクトルID数: {len(vector_ids)}")
    except Exception as e:
        print(f"リスト方式でエラー発生: {e}")
        print("代替方法でベクトルIDを取得します...")
        dummy_vector = [0.0] * dimension
        fetch_limit = min(10000, total_vector_count)
        query_response = index.query(
            vector=dummy_vector,
            top_k=fetch_limit,
            include_metadata=False,
            include_values=False
        )
        vector_ids = [match.id for match in query_response.matches]
        print(f"取得したベクトルID数: {len(vector_ids)}")
    
    vector_ids.sort()
    total_pages = display_page(vector_ids, page_size, page_number)

    while True:
        print("\nコマンド:")
        print("- 次のページを表示: n または next")
        print("- 前のページを表示: p または prev")
        print("- 特定のページに移動: 1-{} の数字".format(total_pages))
        print("- 終了して更新処理へ進む: q または quit")
        
        command = input("\n操作を入力してください: ").lower().strip()
        
        if command in ['q', 'quit']:
            break
        elif command in ['n', 'next']:
            if page_number < total_pages:
                page_number += 1
                display_page(vector_ids, page_size, page_number)
        elif command in ['p', 'prev']:
            if page_number > 1:
                page_number -= 1
                display_page(vector_ids, page_size, page_number)
        else:
            try:
                new_page = int(command)
                if 1 <= new_page <= total_pages:
                    page_number = new_page
                    display_page(vector_ids, page_size, page_number)
                else:
                    print(f"ページ番号は1から{total_pages}の間で指定してください。")
            except ValueError:
                print("無効なコマンドです。")
    
    # 更新処理スタート
    print(f"\n=== type='image' または 'figure_description' のベクトルを weight={new_weight} に更新します ===")
    
    # 更新前に確認
    confirm = input(f"本当に weight を {new_weight} に更新しますか？ (y/N): ")
    if confirm.lower() != 'y':
        print("更新をキャンセルしました。")
        return
    
    batch_size = 100
    ids_to_update = []
    current_metadata = {}

    # 対象ベクトルの特定と現在のメタデータ保存
    for i in range(0, len(vector_ids), batch_size):
        batch_ids = vector_ids[i:i+batch_size]
        try:
            fetched = index.fetch(ids=batch_ids)
            for vid, vec in fetched.vectors.items():
                metadata = vec.metadata
                vec_type = metadata.get("type")
                if vec_type in ["image", "figure_description"]:
                    ids_to_update.append(vid)
                    current_metadata[vid] = metadata.copy()  # 既存メタデータを保存
                    print(f"📋 {vid}: 現在のweight={metadata.get('weight', 'なし')}")
        except Exception as e:
            print(f"フェッチ失敗（{i}-{i+batch_size}）: {e}")

    print(f"\n更新対象: {len(ids_to_update)} 件")
    
    if len(ids_to_update) == 0:
        print("更新対象が見つかりませんでした。")
        return

    # バックアップ情報を表示
    print(f"\n📋 バックアップ情報:")
    for vid in ids_to_update[:5]:  # 最初の5件のみ表示
        old_weight = current_metadata[vid].get('weight', 'なし')
        print(f"  {vid}: {old_weight} → {new_weight}")
    if len(ids_to_update) > 5:
        print(f"  ... 他 {len(ids_to_update) - 5} 件")

    # 更新実行
    success_count = 0
    error_count = 0
    
    for vid in ids_to_update:
        try:
            # 既存のメタデータを保持して重みのみ更新
            updated_metadata = current_metadata[vid].copy()
            updated_metadata["weight"] = new_weight
            
            index.update(id=vid, set_metadata=updated_metadata)
            success_count += 1
            print(f"✅ {vid}: weight → {new_weight} に更新完了")
        except Exception as e:
            error_count += 1
            print(f"❌ {vid}: 更新失敗 - {e}")

    print(f"\n✅ 更新処理完了")
    print(f"成功: {success_count} 件")
    print(f"失敗: {error_count} 件")
    
    if error_count > 0:
        print(f"⚠️  {error_count} 件の更新に失敗しました。ログを確認してください。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PineconeのベクトルIDを表示し、図・画像のweightを更新します')
    parser.add_argument('--page-size', type=int, default=300, help='1ページあたりの表示件数（デフォルト: 300）')
    parser.add_argument('--page', type=int, default=1, help='表示を開始するページ番号（デフォルト: 1）')
    parser.add_argument('--weight', type=float, default=0.7, help='新しいweight値（デフォルト: 0.7）')
    args = parser.parse_args()
    
    get_vector_ids_and_update_weight(args.page_size, args.page, args.weight)