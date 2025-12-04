import os
import re
import glob
import numpy as np
import openai
import torch
import open_clip
from PIL import Image
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import uuid

# 環境変数をロード
load_dotenv()

# ファイルの先頭付近に定数を定義
CATEGORY = "dental"
QDRANT_COLLECTION = "raiden-main"

# CATEGORY = "badminton"
# QDRANT_COLLECTION = "badminton"

@dataclass
class Metadata:
    title: str = ""
    topic: str = ""
    items: List[str] = field(default_factory=list) 
    content: str = ""
    figure_descriptions: Dict[str, str] = field(default_factory=dict)
    case_descriptions: Dict[str, str] = field(default_factory=dict) 
    

@dataclass
class Entry:
    type: str
    number: str
    text: str
    text_id: str
    image_id: str
    metadata: Metadata
    base_name: str = ""

class TextProcessor:
    def __init__(self):
        # 正規表現パターンを修正してFigとcaseの両方に対応
        self.figure_pattern = re.compile(r'\[(Fig\d+[a-zA-Z0-9]*)(?:\]|$)')
        self.case_pattern = re.compile(r'\[(case\d+[a-zA-Z0-9]*)(?:\]|$)')
        self.any_pattern = re.compile(r'\[((Fig|case)\d+[a-zA-Z0-9]*)(?:\]|$)')

    def process_file(self, file_path: str) -> List[Entry]:
        """テキストファイルを処理し、メタデータと図・ケースの説明を抽出"""
        entries = []
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            metadata = Metadata()
            
            # ファイル名からメタデータを抽出（デフォルト値）
            filename = os.path.basename(file_path)
            metadata.title = filename.split('.')[0]
            
            # 行ごとに処理して厳密なタグマッチングを行う
            for line in text.split('\n'):
                line = line.strip()
                
                # 各タグを厳密にパターンマッチ
                if line.startswith('title[') and line.endswith(']'):
                    content = line[6:-1]  # 'title[' と ']' を除去
                    metadata.title = content.strip()
                    print(f"タイトル抽出（行単位）: {metadata.title}")
                
                elif line.startswith('topic[') and line.endswith(']'):
                    content = line[6:-1]  # 'topic[' と ']' を除去
                    metadata.topic = content.strip()
                    print(f"トピック抽出（行単位）: {metadata.topic}")
                
                elif line.startswith('item[') and line.endswith(']'):
                    content = line[5:-1]  # 'item[' と ']' を除去
                    metadata.items.append(content.strip())  # ← 上書きではなくリストに追加
                    print(f"アイテム抽出（行単位）: {content.strip()}")
            
            # 確認のためにメタデータを表示
            print(f"\nメタデータ抽出結果:")
            print(f"タイトル: {metadata.title}")
            print(f"トピック: {metadata.topic}")
            print(f"アイテム: {metadata.items}")
            
            # 本文と図・ケースの説明を分離
            main_content, illustration_content = self._split_content_and_illustrations(text)
            metadata.content = main_content.strip()
            
            # 図・ケースの説明を処理
            if illustration_content:
                self._process_illustration_descriptions(illustration_content, metadata)
            
            # Entryを作成
            entry = Entry(
                type="content",
                number="",
                text=metadata.content,
                text_id=file_path,
                image_id="",
                metadata=metadata,
                base_name=base_name
            )
            entries.append(entry)
            
            # 図の説明文の処理（既存コード）
            for fig_id, description in metadata.figure_descriptions.items():
                if description:
                    entry = Entry(
                        type="figure_description",
                        number=fig_id,
                        text=description,
                        text_id=f"{base_name}_{fig_id}_desc",
                        image_id=f"{base_name}_{fig_id}_image",
                        metadata=metadata,
                        base_name=base_name
                    )
                    entries.append(entry)
                    print(f"図の説明エントリー追加: {fig_id} -> {entry.text_id}")
            
            # ケースの説明文の処理（既存コード）
            for case_id, description in metadata.case_descriptions.items():
                if description:
                    entry = Entry(
                        type="case_description",
                        number=case_id,
                        text=description,
                        text_id=f"{base_name}_{case_id}_desc",
                        image_id=f"{base_name}_{case_id}_image",
                        metadata=metadata,
                        base_name=base_name
                    )
                    entries.append(entry)
                    print(f"ケースの説明エントリー追加: {case_id} -> {entry.text_id}")
            
        except Exception as e:
            print(f"テキスト抽出エラー: {e}")
            import traceback
            traceback.print_exc()  # スタックトレースを表示
        
        return entries

    def _split_content_and_illustrations(self, text: str) -> Tuple[str, str]:
        """本文と図・ケースの説明を分離"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Fig または case のいずれかのパターンにマッチするか確認
            if self.any_pattern.match(line.strip()):
                return '\n'.join(lines[:i]).strip(), '\n'.join(lines[i:])
        return text.strip(), ""

    def _process_illustration_descriptions(self, illustration_content: str, metadata: Metadata):
        """図・ケースの説明文を処理"""
        current_id = None
        current_text = []
        current_type = None  # "figure" または "case"
        
        for line in illustration_content.split('\n'):
            # Figパターンのチェック
            fig_match = self.figure_pattern.match(line.strip())
            # caseパターンのチェック
            case_match = self.case_pattern.match(line.strip())
            
            if fig_match:
                # 前のエントリーを保存
                if current_id and current_text:
                    if current_type == "figure":
                        metadata.figure_descriptions[current_id] = '\n'.join(current_text).strip()
                    elif current_type == "case":
                        metadata.case_descriptions[current_id] = '\n'.join(current_text).strip()
                
                current_id = fig_match.group(1)
                current_text = []
                current_type = "figure"
                # デバッグ情報を出力
                print(f"図説明を検出: {current_id}")
            elif case_match:
                # 前のエントリーを保存
                if current_id and current_text:
                    if current_type == "figure":
                        metadata.figure_descriptions[current_id] = '\n'.join(current_text).strip()
                    elif current_type == "case":
                        metadata.case_descriptions[current_id] = '\n'.join(current_text).strip()
                
                current_id = case_match.group(1)
                current_text = []
                current_type = "case"
                # デバッグ情報を出力
                print(f"ケース説明を検出: {current_id}")
            elif line.strip() and current_id:
                current_text.append(line.strip())
        
        # 最後のエントリーを保存
        if current_id and current_text:
            if current_type == "figure":
                metadata.figure_descriptions[current_id] = '\n'.join(current_text).strip()
            elif current_type == "case":
                metadata.case_descriptions[current_id] = '\n'.join(current_text).strip()

class ImageProcessor:
    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device

    def get_embedding(self, image_path: str) -> Dict:
        """画像のembeddingを生成"""
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_embedding = self.model.encode_image(image)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            vector = image_embedding.cpu().numpy().tolist()[0]
            expanded_vector = np.concatenate([vector, np.zeros(1536-len(vector))]).tolist()
            return {
                "vector": expanded_vector,
                "status": "success"
            }
        except Exception as e:
            print(f"画像エンベディングエラー: {e}")
            return {
                "vector": [0.0] * 1536,
                "status": "error",
                "error_message": str(e)
            }

class TextEmbeddingProcessor:
    def __init__(self, client):
        self.client = client
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )

    def get_embedding(self, text: str) -> List[float]:
        """テキストのembeddingを生成"""
        chunks = self.text_splitter.split_text(text)
        embedding_sum = np.zeros(1536)
        count = 0
        
        for chunk in chunks:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
                embedding_sum += np.array(response.data[0].embedding)
                count += 1
            except Exception as e:
                print(f"Embedding生成エラー: {e}")
        
        return (embedding_sum / count if count > 0 else embedding_sum).tolist()

def validate_environment():
    """環境変数の検証"""
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY")
    }
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise EnvironmentError(f"必要な環境変数が設定されていません: {', '.join(missing_vars)}")
    return required_vars

def initialize_services():
    """サービスの初期化"""
    env_vars = validate_environment()
    
    # OpenAI クライアント
    openai_client = openai.OpenAI(api_key=env_vars["OPENAI_API_KEY"])
    
    # Qdrant クライアント
    qdrant_client = QdrantClient(
        url=env_vars["QDRANT_URL"],
        api_key=env_vars["QDRANT_API_KEY"]
    )
    
    # コレクションの存在確認と作成
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if QDRANT_COLLECTION not in collection_names:
            print(f"\nコレクション '{QDRANT_COLLECTION}' を作成します...")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )
            print(f"コレクション '{QDRANT_COLLECTION}' を作成しました")
        else:
            print(f"\nコレクション '{QDRANT_COLLECTION}' は既に存在します")
    except Exception as e:
        print(f"コレクション確認/作成エラー: {e}")
        raise
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
    model.to(device)
    
    return openai_client, qdrant_client, model, preprocess, device

def create_points(data: Entry, related_ids: Dict[str, List[str]] = None) -> List[PointStruct]:
    """Qdrant用のPointStructを作成"""
    points = []
    content_id = f"{data.base_name}_content"
    
    # タイプごとの重み設定
    type_weights = {
        "content": 1.0,
        "figure_description": 0.7,
        "case_description": 0.7,
        "image": 0.7
    }
    
    if data.type == "content":
        related_images = []
        related_descriptions = []
        related_cases = []
        
        for fig_id in data.metadata.figure_descriptions.keys():
            related_images.append(f"{data.base_name}_{fig_id}_image")
            related_descriptions.append(f"{data.base_name}_{fig_id}_desc")
        
        for case_id in data.metadata.case_descriptions.keys():
            related_cases.append(f"{data.base_name}_{case_id}_desc")
            related_images.append(f"{data.base_name}_{case_id}_image")
        
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=data.text_embedding,
            payload={
                # 全てトップレベルに配置
                "text": data.text,                    # ← page_contentではなくtext
                "category": CATEGORY,
                "title": data.metadata.title,
                "topic": data.metadata.topic,
                "items": data.metadata.items,
                "type": data.type,                    # ← トップレベル！
                "weight": type_weights["content"],    # ← トップレベル！
                "vector_id": content_id,
                "original_id": content_id,            # ← 追加
                "related_images": related_images,
                "related_descriptions": related_descriptions,
                "related_cases": related_cases
            }
        ))
        
    elif data.type == "figure_description":
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=data.text_embedding,
            payload={
                "text": data.text,
                "category": CATEGORY,
                "title": data.metadata.title,
                "topic": data.metadata.topic,
                "items": data.metadata.items,
                "type": data.type,                           # ← トップレベル！
                "weight": type_weights["figure_description"], # ← トップレベル！
                "vector_id": data.text_id,
                "original_id": data.text_id,                  # ← 追加
                "related_content_id": content_id,
                "related_image_id": data.image_id
            }
        ))
        
    elif data.type == "case_description":
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=data.text_embedding,
            payload={
                "text": data.text,
                "category": CATEGORY,
                "title": data.metadata.title,
                "topic": data.metadata.topic,
                "items": data.metadata.items,
                "type": data.type,                          # ← トップレベル！
                "weight": type_weights["case_description"], # ← トップレベル！
                "vector_id": data.text_id,
                "original_id": data.text_id,                # ← 追加
                "related_content_id": content_id,
                "related_image_id": data.image_id
            }
        ))
        
    elif data.type == "image":
        image_text_parts = []
        if data.metadata.title:
            image_text_parts.append(data.metadata.title)
        if data.metadata.topic:
            image_text_parts.append(data.metadata.topic)
        if data.metadata.items:
            image_text_parts.extend(data.metadata.items)
        
        image_text = " ".join(image_text_parts)
        
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=data.image_embedding,
            payload={
                "text": image_text,
                "category": CATEGORY,
                "title": data.metadata.title,
                "topic": data.metadata.topic,
                "items": data.metadata.items,
                "type": data.type,                # ← トップレベル！
                "weight": type_weights["image"],  # ← トップレベル！
                "vector_id": data.image_id,
                "original_id": data.image_id,     # ← 追加
                "related_content_id": content_id,
                "related_description_id": data.text_id
            }
        ))
    
    return points

def main():
    try:
        # サービスの初期化
        openai_client, qdrant_client, model, preprocess, device = initialize_services()
        
        # プロセッサーの初期化
        text_processor = TextProcessor()
        image_processor = ImageProcessor(model, preprocess, device)
        text_embedding_processor = TextEmbeddingProcessor(openai_client)
        
        # テキストファイルの処理
        txt_files = glob.glob("data/*.txt")
        if not txt_files:
            raise FileNotFoundError("data ディレクトリにtxtファイルが見つかりません")

        # 本文と画像説明の抽出
        all_entries = []
        for file_path in txt_files:
            print(f"\nテキストファイル処理: {file_path}")
            entries = text_processor.process_file(file_path)
            all_entries.extend(entries)

        # 画像ファイルの確認
        image_files = glob.glob(os.path.join("data", "Fig*.jpg"))
        case_image_files = glob.glob(os.path.join("data", "case*.jpg"))  # case画像ファイルも確認
        
        # デバッグ: 抽出されたすべての図とケースを表示
        print("\n== 抽出された図の説明 ==")
        for entry in all_entries:
            if entry.type == "figure_description":
                print(f"ID: {entry.number}, Text ID: {entry.text_id}, Image ID: {entry.image_id}")
                
        print("\n== 抽出されたケースの説明 ==")
        for entry in all_entries:
            if entry.type == "case_description":
                print(f"ID: {entry.number}, Text ID: {entry.text_id}, Image ID: {entry.image_id}")

        # 抽出結果の表示と確認
        print("\n=== 抽出結果の確認 ===")
        content_count = sum(1 for e in all_entries if e.type == "content")
        fig_desc_count = sum(1 for e in all_entries if e.type == "figure_description")
        case_desc_count = sum(1 for e in all_entries if e.type == "case_description")
        print(f"メインテキスト数: {content_count}")
        print(f"画像説明テキスト数: {fig_desc_count}")
        print(f"ケース説明テキスト数: {case_desc_count}")
        print(f"Fig画像ファイル数: {len(image_files)}")
        print(f"ケース画像ファイル数: {len(case_image_files)}")

        print("\n画像テキストと画像ファイルの対応:")
        print("\n画像テキスト:")
        for entry in all_entries:
            if entry.type == "figure_description":
                print(f"- {entry.text_id} -> {entry.image_id}")
        
        print("\nケーステキスト:")
        for entry in all_entries:
            if entry.type == "case_description":
                print(f"- {entry.text_id} -> {entry.image_id}")

        print("\nFig画像ファイル:")
        for img in sorted(image_files):
            print(f"- {os.path.basename(img)}")
            
        print("\nケース画像ファイル:")
        for img in sorted(case_image_files):
            print(f"- {os.path.basename(img)}")

        # 処理開始の確認
        confirm = input("\n上記のデータに対してembedding処理を開始しますか？ (y/n): ").lower().strip()
        if confirm != 'y':
            print("処理を中止します")
            return

        # embedding処理
        points_to_upsert = []
        
        # 各ファイルの処理
        for file_path in txt_files:
            print(f"\n処理中のファイル: {file_path}")
            
            # メインテキストと図・ケースの説明の処理
            entries = text_processor.process_file(file_path)
            for entry in entries:
                entry.text_embedding = text_embedding_processor.get_embedding(entry.text)
                points_to_upsert.extend(create_points(entry))
            
            # Fig画像の処理
            fig_pattern = os.path.join("data", "Fig*.jpg")
            for img_path in glob.glob(fig_pattern):
                img_name = os.path.basename(img_path)
                # 正規表現をより正確に - 完全な Fig1a などの形式にマッチ
                match = re.match(r'(Fig\d+[a-zA-Z0-9]*)\.jpg', img_name)
                if match:
                    fig_id = match.group(1)  # 完全なFig IDを取得 (例: Fig1a)
                    image_id = f"{entries[0].base_name}_{fig_id}_image"
                    
                    image_result = image_processor.get_embedding(img_path)
                    if image_result["status"] == "success":
                        image_entry = Entry(
                            type="image",
                            number=fig_id,  # 完全なIDを使用
                            text="",
                            text_id=f"{entries[0].base_name}_{fig_id}_desc",  # 完全なIDで説明と一致
                            image_id=image_id,
                            metadata=entries[0].metadata,
                            base_name=entries[0].base_name
                        )
                        image_entry.image_embedding = image_result["vector"]
                        points_to_upsert.extend(create_points(image_entry))
                        print(f"Fig画像のembedding生成完了: {image_id}")
                    else:
                        print(f"画像の処理に失敗: {img_path}")
            
            # case画像の処理
            case_pattern = os.path.join("data", "case*.jpg")
            for img_path in glob.glob(case_pattern):
                img_name = os.path.basename(img_path)
                match = re.match(r'(case\d+[a-zA-Z0-9]*)\.jpg', img_name)
                if match:
                    case_id = match.group(1)  # 例: "case1a1"
                    image_id = f"{entries[0].base_name}_{case_id}_image"
                    text_id  = f"{entries[0].base_name}_{case_id}_desc"

                    image_result = image_processor.get_embedding(img_path)
                    if image_result["status"] == "success":
                        image_entry = Entry(
                            type="image",
                            number=case_id,
                            text="",
                            text_id=text_id,
                            image_id=image_id,
                            metadata=entries[0].metadata,
                            base_name=entries[0].base_name
                        )
                        image_entry.image_embedding = image_result["vector"]
                        points_to_upsert.extend(create_points(image_entry))
                        print(f"ケース画像のembedding生成完了: {image_id}")
                    else:
                        print(f"画像の処理に失敗: {img_path}")
                
        # 処理結果の表示
        print("\n=== 処理結果 ===")
        print(f"処理したポイント数: {len(points_to_upsert)}")
        print("\n各データの重みと内訳:")
        
        # 型別のカウントと重みの表示
        type_counts = {}
        for p in points_to_upsert:
            p_type = p.payload["metadata"]["type"]      # 変更！
            weight = p.payload["metadata"]["weight"]     # 変更！
            if p_type not in type_counts:
                type_counts[p_type] = {
                    "count": 0,
                    "weight": weight
                }
            type_counts[p_type]["count"] += 1
            
        for t, info in type_counts.items():
            print(f"- {t}:")
            print(f"  - 件数: {info['count']}")
            print(f"  - 重み: {info['weight']}")

        # Qdrantへのアップロード
        if points_to_upsert:
            try:
                # バッチでアップロード（Qdrantは一度に大量のポイントを処理できる）
                batch_size = 100
                for i in range(0, len(points_to_upsert), batch_size):
                    batch = points_to_upsert[i:i+batch_size]
                    qdrant_client.upsert(
                        collection_name=QDRANT_COLLECTION,
                        points=batch
                    )
                    print(f"進捗: {min(i+batch_size, len(points_to_upsert))}/{len(points_to_upsert)} ポイントをアップロード")
                
                print(f"\n✅ {len(points_to_upsert)}個のポイントをQdrantに保存しました")
                
                # コレクション情報の確認
                collection_info = qdrant_client.get_collection(QDRANT_COLLECTION)
                print(f"\nコレクション情報:")
                print(f"- 総ポイント数: {collection_info.points_count}")
                # print(f"- ベクトル次元: {collection_info.config.params.vectors.size}")
                
            except Exception as e:
                print(f"\nQdrantへの保存中にエラーが発生: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nアップロードするデータがありません")

    except Exception as e:
        print(f"\n処理中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()