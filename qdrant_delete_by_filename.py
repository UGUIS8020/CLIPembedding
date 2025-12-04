import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rich.console import Console
from rich.table import Table
from rich import print as rprint

load_dotenv()

# 環境変数からURLとAPIキーを取得
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "raiden-main"

def initialize_qdrant():
    """Qdrantの初期化を行う"""
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=180
        )
        
        # 接続テスト
        collections = client.get_collections()
        print(f"✅ Qdrant接続成功。コレクション数: {len(collections.collections)}")
        return client
    except Exception as e:
        rprint(f"[red]Qdrantの初期化でエラーが発生しました: {e}[/red]")
        raise

def search_by_filename(client, filename):
    """ファイル名で検索して該当するポイントを取得"""
    try:
        rprint(f"[yellow]🔍 '{filename}' を検索中...[/yellow]")
        
        # 全データを取得してファイル名で部分一致検索
        all_results, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10000  # 大きな値で全件取得
        )
        
        # クライアント側で部分一致フィルタリング
        matching_points = []
        for point in all_results:
            payload = point.payload
            # title, original_id, text などのフィールドでファイル名を検索
            for field in ['title', 'original_id', 'text']:
                if field in payload and payload[field]:
                    field_value = str(payload[field]).lower()
                    if filename.lower() in field_value:
                        matching_points.append(point)
                        break
        
        rprint(f"[green]✅ {len(matching_points)} 件のマッチするデータが見つかりました[/green]")
        return matching_points
        
    except Exception as e:
        rprint(f"[red]検索でエラーが発生しました: {e}[/red]")
        return []

def display_search_results(points, console):
    """検索結果を表示"""
    if not points:
        return
    
    rprint(f"\n[bold blue]📋 検索結果一覧:[/bold blue]")
    
    for i, point in enumerate(points, 1):
        rprint(f"\n[bold yellow]📎 結果 {i}: ID {point.id}[/bold yellow]")
        
        # 主要な情報のみ表示
        payload = point.payload
        important_fields = ['title', 'original_id', 'type', 'category', 'topic']
        
        table = Table(show_header=True, header_style="bold magenta", width=100)
        table.add_column("フィールド", style="dim", width=15)
        table.add_column("値", style="yellow", width=80)
        
        for field in important_fields:
            if field in payload:
                value = str(payload[field])
                if len(value) > 100:
                    value = value[:100] + "..."
                table.add_row(field, value)
        
        # textフィールドは最初の100文字のみ表示
        if 'text' in payload:
            text_preview = str(payload['text'])[:100] + "..." if len(str(payload['text'])) > 100 else str(payload['text'])
            table.add_row("text (preview)", text_preview)
        
        console.print(table)

def delete_points_by_ids(client, point_ids):
    """指定されたIDのポイントを削除"""
    try:
        if not point_ids:
            rprint("[yellow]削除対象のデータがありません[/yellow]")
            return False
        
        rprint(f"[yellow]🗑️  {len(point_ids)} 件のデータを削除中...[/yellow]")
        
        # バッチ削除
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=point_ids
        )
        
        rprint(f"[green]✅ {len(point_ids)} 件のデータを正常に削除しました[/green]")
        return True
        
    except Exception as e:
        rprint(f"[red]削除でエラーが発生しました: {e}[/red]")
        return False

def search_and_delete_by_filename(client, console, filename):
    """ファイル名で検索して削除"""
    try:
        # 1. 検索実行
        matching_points = search_by_filename(client, filename)
        
        if not matching_points:
            rprint(f"[red]⚠️ '{filename}' に関連するデータが見つかりませんでした[/red]")
            return
        
        # 2. 検索結果を表示
        display_search_results(matching_points, console)
        
        # 3. 削除確認
        rprint(f"\n[bold red]⚠️  警告: {len(matching_points)} 件のデータが削除されます！[/bold red]")
        confirm = input("本当に削除しますか？ (yes/no): ").strip().lower()
        
        if confirm in ['yes', 'y', 'はい']:
            # 4. 削除実行
            point_ids = [point.id for point in matching_points]
            success = delete_points_by_ids(client, point_ids)
            
            if success:
                # 5. 削除後の確認
                rprint("\n[bold green]🔍 削除後の確認検索を実行中...[/bold green]")
                remaining_points = search_by_filename(client, filename)
                if not remaining_points:
                    rprint("[green]✅ 削除が完了しました。該当データは見つかりません。[/green]")
                else:
                    rprint(f"[yellow]⚠️ まだ {len(remaining_points)} 件の関連データが残っています[/yellow]")
        else:
            rprint("[yellow]削除をキャンセルしました[/yellow]")
        
    except Exception as e:
        rprint(f"[red]処理でエラーが発生しました: {e}[/red]")

def main():
    """メイン処理"""
    try:
        console = Console()
        client = initialize_qdrant()
        
        while True:
            rprint("\n[bold cyan]🗑️  ファイル名削除ツール[/bold cyan]")
            filename = input("削除対象のファイル名を入力してください (例: Transplantation_quint201901): ").strip()
            
            if not filename:
                rprint("[yellow]ファイル名が入力されていません。終了します。[/yellow]")
                break
            
            if filename.lower() in ['exit', 'quit', 'q']:
                break
            
            search_and_delete_by_filename(client, console, filename)
            
            # 続行確認
            continue_choice = input("\n他のファイルも削除しますか？ (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', 'はい']:
                break
        
        rprint("[bold green]👋 削除ツールを終了します[/bold green]")
        
    except Exception as e:
        rprint(f"[red]エラーが発生しました: {e}[/red]")
        return

if __name__ == "__main__":
    main()