import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import json

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

def format_metadata(metadata):
    """メタデータを見やすく整形する"""
    formatted = {}
    for key, value in metadata.items():
        if isinstance(value, str) and len(value) > 100:
            # テキストが長い場合は省略
            formatted[key] = value[:100] + "..."
        else:
            formatted[key] = value
    return formatted

def display_search_results(results, console, search_term=""):
    """検索結果を表示する"""
    if not results:
        rprint(f"[red]⚠️ '{search_term}' に関連するデータが見つかりませんでした[/red]")
        return
    
    rprint(f"\n[bold green]🔍 検索結果: {len(results)}件 (検索語: '{search_term}')[/bold green]")
    
    for i, point in enumerate(results, 1):
        rprint(f"\n[bold blue]📎 結果 {i}: ID {point.id}[/bold blue]")
        
        # メタデータをテーブルで表示
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("キー", style="dim")
        table.add_column("値", style="yellow")
        
        formatted_metadata = format_metadata(point.payload)
        for key, value in formatted_metadata.items():
            # 長いテキストは改行して表示
            if isinstance(value, str):
                value = value.replace('\n', ' ')
            table.add_row(key, str(value))
        
        console.print(table)

def search_by_filename(client, console, filename):
    """ファイル名で検索"""
    try:
        # ファイル名の部分一致検索
        # titleフィールドで検索
        filter_condition = Filter(
            should=[
                FieldCondition(
                    key="title",
                    match=MatchValue(value=filename)
                ),
                FieldCondition(
                    key="original_id", 
                    match=MatchValue(value=filename)
                )
            ]
        )
        
        # スクロール検索で全件取得
        results, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=filter_condition,
            limit=100
        )
        
        if not results:
            # 部分一致で再検索
            rprint(f"[yellow]完全一致が見つからないため、部分一致で検索中...[/yellow]")
            
            # 全データを取得して部分一致検索
            all_results, _ = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=10000  # 大きな値で全件取得
            )
            
            # クライアント側で部分一致フィルタリング
            results = []
            for point in all_results:
                payload = point.payload
                # title, original_id, text などのフィールドで部分一致検索
                for field in ['title', 'original_id', 'text']:
                    if field in payload and payload[field]:
                        field_value = str(payload[field]).lower()
                        if filename.lower() in field_value:
                            results.append(point)
                            break
        
        display_search_results(results, console, filename)
        return results
        
    except Exception as e:
        rprint(f"[red]検索でエラーが発生しました: {e}[/red]")
        return []

def search_by_content(client, console, content):
    """テキスト内容で検索"""
    try:
        # 全データを取得してテキスト内容で検索
        all_results, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10000  # 大きな値で全件取得
        )
        
        # クライアント側でテキスト内容検索
        results = []
        for point in all_results:
            payload = point.payload
            if 'text' in payload and payload['text']:
                text_content = str(payload['text']).lower()
                if content.lower() in text_content:
                    results.append(point)
        
        display_search_results(results, console, content)
        return results
        
    except Exception as e:
        rprint(f"[red]テキスト検索でエラーが発生しました: {e}[/red]")
        return []

def list_all_data(client, console):
    """全データを表示"""
    try:
        # コレクション情報を取得
        collection_info = client.get_collection(QDRANT_COLLECTION)
        total_count = collection_info.points_count
        
        rprint(f"\n[bold green]📊 コレクション '{QDRANT_COLLECTION}' の総ベクトル数: {total_count}[/bold green]")
        
        # 最初の10件を表示
        results, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10
        )
        
        rprint(f"\n[bold]🔍 最初の10件のデータ:[/bold]")
        display_search_results(results, console, "全データ(最初の10件)")
        
    except Exception as e:
        rprint(f"[red]データ取得でエラーが発生しました: {e}[/red]")

def main():
    """メイン処理"""
    try:
        console = Console()
        client = initialize_qdrant()
        
        while True:
            rprint("\n[bold cyan]🔍 Qdrant検索メニュー[/bold cyan]")
            rprint("1. ファイル名で検索")
            rprint("2. テキスト内容で検索") 
            rprint("3. 全データ表示(最初の10件)")
            rprint("4. 終了")
            
            choice = input("\n選択してください (1-4): ").strip()
            
            if choice == "1":
                filename = input("検索するファイル名を入力してください: ").strip()
                if filename:
                    search_by_filename(client, console, filename)
            
            elif choice == "2":
                content = input("検索するテキスト内容を入力してください: ").strip()
                if content:
                    search_by_content(client, console, content)
            
            elif choice == "3":
                list_all_data(client, console)
            
            elif choice == "4":
                rprint("[bold green]👋 検索を終了します[/bold green]")
                break
            
            else:
                rprint("[red]無効な選択です。1-4を入力してください。[/red]")
        
    except Exception as e:
        rprint(f"[red]エラーが発生しました: {e}[/red]")
        return

if __name__ == "__main__":
    main()