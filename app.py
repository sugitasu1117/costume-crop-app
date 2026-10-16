import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import os
import urllib.request
from ultralytics import YOLO

st.set_page_config(page_title="画像領域処理アプリ", layout="wide")

st.title("✂️ 画像領域処理（抽出 / 黒塗り）アプリ")
st.write("対象（顔・衣装）と処理方法（切り出し・黒塗り）を選択して画像をアップロードしてください。")

# モデル・検出器の読み込み（キャッシュ化して高速化）
@st.cache_resource
def load_models():
    # 1. 人物/全身検出用 (yolov8n.pt は ultralytics が自動取得)
    person_model = YOLO("yolov8n.pt")
    
    # 2. 顔検出用 (yolov8n-face.pt を Hugging Face から安全にローカル取得)
    face_model_path = "yolov8n-face.pt"
    if not os.path.exists(face_model_path) or os.path.getsize(face_model_path) == 0:
        url = "https://huggingface.co/Bingsu/aDetailer/resolve/main/yolov8n-face.pt"
        # ユーザーエージェントを設定してダウンロードの拒否を回避
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(face_model_path, 'wb') as out_file:
            out_file.write(response.read())
            
    face_model = YOLO(face_model_path)
    return person_model, face_model

person_model, face_model = load_models()

# サイドバーで機能選択
st.sidebar.header("⚙️ 設定")
target_type = st.sidebar.radio("1. 対象を選択", ["顔 (Face)", "衣装/全身 (Costume/Person)"])
action_type = st.sidebar.radio("2. 処理方法を選択", ["トリミング切り出し (Crop)", "黒塗りマスキング (Mask)"])

uploaded_files = st.file_uploader("画像をアップロード（複数可）", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

def detect_regions(img_np, target):
    """選択された対象の座標バウンディングボックス (x, y, w, h) リストを返す"""
    boxes = []
    if "顔" in target:
        results = face_model(img_np, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2 - x1, y2 - y1))
    else:
        results = person_model(img_np, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # 0: person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

if uploaded_files:
    processed_images = []
    
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        
        boxes = detect_regions(img_np, target_type)
        
        if "トリミング" in action_type:
            # 切り出し処理
            for i, (x, y, w, h) in enumerate(boxes):
                cropped = img_np[y:y+h, x:x+w]
                crop_pil = Image.fromarray(cropped)
                file_name = f"{uploaded_file.name.split('.')[0]}_crop_{i+1}.png"
                processed_images.append((file_name, crop_pil))
        else:
            # 黒塗り処理
            masked_np = img_np.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(masked_np, (x, y), (x + w, y + h), (0, 0, 0), -1)
            
            masked_pil = Image.fromarray(masked_np)
            file_name = f"{uploaded_file.name.split('.')[0]}_masked.png"
            processed_images.append((file_name, masked_pil))

    if processed_images:
        st.subheader("🖼️ 処理結果プレビュー")
        cols = st.columns(min(len(processed_images), 4))
        for idx, (name, img) in enumerate(processed_images):
            with cols[idx % 4]:
                st.image(img, caption=name, use_container_width=True)

        # ZIPファイル作成
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for name, img in processed_images:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                zf.writestr(name, img_byte_arr.getvalue())
        
        st.success(f"合計 {len(processed_images)} 件の処理が完了しました！")
        st.download_button(
            label="📦 処理済み画像をZIPで一括ダウンロード",
            data=zip_buffer.getvalue(),
            file_name="processed_images.zip",
            mime="application/zip"
        )
    else:
        st.warning("指定された対象が画像から検出されませんでした。")