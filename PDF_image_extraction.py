import os
from tkinter import Tk, filedialog
import fitz 
import cv2

# 出力用フォルダ
PAGES_DIR = "pages"      # PDF→ページ画像
FIGURES_DIR = "figures"  # ページ画像→図を切り出し


import numpy as np  # まだ import していなければ追加

def deskew_image(img):
    """ページ全体の傾きを推定して補正する（ごくわずかな傾きだけ補正）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(
        gray_inv, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img, 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # OpenCV仕様の角度補正
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # --- ここから制限ロジック ---

    # ほぼ0度なら何もしない
    if abs(angle) < 0.05:  # 0.05°未満は誤差として無視
        print(f"  [deskew] angle ≒ 0 -> skip")
        return img, 0.0

    # ±2°を超える大きな角度は「誤判定」とみなして補正しない
    if abs(angle) > 2.0:
        print(f"  [deskew] angle = {angle:.3f} deg -> skip (>2°)")
        return img, 0.0

    # 実際に回転をかけるのは ±0.05°〜±2° の範囲だけ
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    print(f"  [deskew] angle = {angle:.3f} deg -> applied")
    return rotated, angle

def pdf_to_pages(pdf_path, dpi=300):
    """PDF をページごとの PNG 画像に変換して PAGES_DIR に保存（PyMuPDF版）"""
    os.makedirs(PAGES_DIR, exist_ok=True)
    print(f"[INFO] PDF からページ画像に変換中: {pdf_path}")

    doc = fitz.open(pdf_path)
    paths = []

    # PyMuPDF はデフォルト 72dpi のため、倍率を掛けて指定dpiにする
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(PAGES_DIR, f"page_{i:03}.png")
        pix.save(out_path)
        print("  saved page image:", out_path)
        paths.append(out_path)

    doc.close()
    return paths


def extract_figures_from_page(page_path, page_no):
    """1ページ画像から図っぽい領域を抽出して FIGURES_DIR に保存"""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    img = cv2.imread(page_path)
    if img is None:
        print(f"[WARN] 画像を読み込めませんでした: {page_path}")
        return

    # ★ ここで一度ページ全体の傾きを補正する
    img, angle = deskew_image(img)

    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # （この下は今のコードのままでOK）
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    idx = 1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 20000:
            continue
        ratio = w / h
        if ratio < 0.4 or ratio > 3.0:
            continue

        crop = orig[y:y + h, x:x + w]

        # ★ 抽出後に上下左右 2px ずつトリミングする
        trim = 2
        ch, cw = crop.shape[:2]

        # 画像サイズが十分に大きい場合のみトリミング
        if ch > 2 * trim and cw > 2 * trim:
            crop = crop[trim:ch - trim, trim:cw - trim]
        else:
            print(f"    [WARN] 画像が小さすぎるためトリミングをスキップしました (size={cw}x{ch})")

        out_path = os.path.join(
            FIGURES_DIR,
            f"page{page_no:03}_fig{idx:02}.jpg"
        )
        cv2.imwrite(out_path, crop)
        print("    saved figure:", out_path)
        idx += 1


def main():
    # スクリプトのある場所
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[INFO] スクリプトの場所: {script_dir}")

    # 前回使ったフォルダを格納するファイル
    last_dir_file = os.path.join(script_dir, "last_dir.txt")

    # 初期フォルダを決定
    initial_dir = script_dir
    if os.path.exists(last_dir_file):
        try:
            with open(last_dir_file, "r", encoding="utf-8") as f:
                saved_dir = f.read().strip()
            if saved_dir and os.path.isdir(saved_dir):
                initial_dir = saved_dir
        except Exception:
            pass

    # 1) PDF をダイアログで選択
    root = Tk()
    root.withdraw()

    pdf_path = filedialog.askopenfilename(
        title="図を抽出したいPDFを選択してください",
        initialdir=initial_dir,  # ← 前回のフォルダ or スクリプトの場所
        filetypes=[
            ("PDF files", ("*.pdf", "*.PDF")),
            ("All files", "*.*"),
        ],
    )

    if not pdf_path:
        print("PDF が選択されませんでした。終了します。")
        return

    print(f"[INFO] 選択されたPDF: {pdf_path}")

    # ★ PDFと同じ場所に「PDF名のフォルダ」を作り、その中に pages / figures を作る
    base_dir = os.path.dirname(pdf_path)
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]  # 拡張子なしファイル名
    output_root = os.path.join(base_dir, pdf_stem)

    # グローバル変数を書き換えて、このPDF専用の出力先にする
    global PAGES_DIR, FIGURES_DIR
    PAGES_DIR = os.path.join(output_root, "pages")
    FIGURES_DIR = os.path.join(output_root, "figures")

    os.makedirs(PAGES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"[INFO] 出力フォルダ: {output_root}")

    # 選択されたPDFのフォルダを次回用に保存
    try:
        current_dir = os.path.dirname(pdf_path)
        with open(last_dir_file, "w", encoding="utf-8") as f:
            f.write(current_dir)
    except Exception as e:
        print(f"[WARN] 前回フォルダの保存に失敗しました: {e}")

    # 2) PDF→ページ画像
    page_paths = pdf_to_pages(pdf_path, dpi=300)

    # 3) ページ画像から図を抽出
    print("[INFO] 図の抽出を開始します。")
    for i, page_path in enumerate(page_paths, start=1):
        print("  processing:", page_path)
        extract_figures_from_page(page_path, i)

    print("[INFO] 完了しました。FIGURES_DIR を確認してください。:", FIGURES_DIR)


if __name__ == "__main__":
    main()