import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile

st.title("👤 顔データ抽出アプリ")
st.write("画像をアップロードすると、検出された顔領域のみをクロップしてZIPでまとめてダウンロードできます。")

# OpenCV同梱の顔検出分類器をロード
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

uploaded_files = st.file_uploader("画像をアップロード（複数可）", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

if uploaded_files:
    cropped_faces = []
    
    for uploaded_file in uploaded_files:
        # 画像の読み込み
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 顔の検出
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for i, (x, y, w, h) in enumerate(faces):
            # 顔領域の切り出し
            face_img = img_np[y:y+h, x:x+w]
            face_pil = Image.fromarray(face_img)
            
            file_name = f"{uploaded_file.name.split('.')[0]}_face_{i+1}.png"
            cropped_faces.append((file_name, face_pil))
            
            # 画面上にプレビュー表示
            st.image(face_pil, caption=f"検出: {file_name}", width=150)

    if cropped_faces:
        # ZIPファイルの作成
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for name, img in cropped_faces:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                zf.writestr(name, img_byte_arr.getvalue())
        
        st.success(f"合計 {len(cropped_faces)} 件の顔領域を抽出しました！")
        st.download_button(
            label="📦 抽出した顔画像をZIPで一括ダウンロード",
            data=zip_buffer.getvalue(),
            file_name="extracted_faces.zip",
            mime="application/zip"
        )
    else:
        st.warning("アップロードされた画像から顔が検出されませんでした。")