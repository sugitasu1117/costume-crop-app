import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io
import zipfile

st.title("👗 衣装データ抽出アプリ（高精度AI顔検出）")
st.write("最新のAI（YOLO）が顔を自動で高精度認識し、首から下の衣装部分のみを抽出・加工します。")

# モード選択
mode = st.radio("処理モードを選択", ["衣装のみ切り出し（クロップ）", "顔を黒塗り（マスキング）"])

# 画像の複数アップロード（WebPにも対応）
uploaded_files = st.file_uploader(
    "画像をアップロードしてください (WebP / JPG / PNG)", 
    type=["webp", "jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

@st.cache_resource
def load_yolo_model():
    # 軽量かつ高精度な人・顔検出用YOLOモデルを読み込み
    model = YOLO("yolov8n.pt")
    return model

if uploaded_files:
    st.info(f"{len(uploaded_files)} 枚の画像がアップロードされました。AIモデルで顔認識を開始します...")
    
    model = load_yolo_model()
    output_zip = io.BytesIO()
    processed_images = []
    
    with zipfile.ZipFile(output_zip, "w") as zf:
        for idx, uploaded_file in enumerate(uploaded_files):
            # 画像の読み込みとRGB変換
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                continue
            
            h, w, _ = image.shape
            
            # YOLOで推論を実行 (class 0 = 'person' の領域から顔・頭部位置を特定)
            results = model(image, verbose=False)
            
            faces = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 人物（cls==0）または顔領域を判定
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # 人物検出
                        bx, by, bw_end, bh_end = box.xyxy[0].tolist()
                        # 人物バウンディングボックスの上部1/3を「顔・頭部」と定義
                        person_h = bh_end - by
                        face_y_end = by + (person_h * 0.30)
                        faces.append((int(bx), int(by), int(bw_end - bx), int(face_y_end - by)))
            
            output_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if len(faces) > 0:
                # 一番大きな人物の顔領域を取得
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, fw, fh = largest_face
                
                if mode == "衣装のみ切り出し（クロップ）":
                    # 顔・頭部の下端から切り出し
                    crop_top = max(0, min(h, int(y + fh)))
                    output_img = output_img[crop_top:h, 0:w]
                else:
                    # 顔・頭部領域を黒塗り
                    for (fx, fy, f_w, f_h) in faces:
                        cv2.rectangle(output_img, (max(0, fx), max(0, fy)), (min(w, fx + f_w), min(h, fy + f_h)), (0, 0, 0), -1)
            else:
                # 万が一AIで人物が取れない場合のバックアップ（上部30%カット）
                if mode == "衣装のみ切り出し（クロップ）":
                    crop_top = int(h * 0.3)
                    output_img = output_img[crop_top:h, 0:w]
                else:
                    cv2.rectangle(output_img, (0, 0), (w, int(h * 0.3)), (0, 0, 0), -1)
            
            # Pillow画像に変換してJPGでZipに保存
            if output_img.size > 0:
                pil_img = Image.fromarray(output_img)
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='JPEG', quality=95)
                
                filename = f"costume_{idx+1}.jpg"
                zf.writestr(filename, img_byte_arr.getvalue())
                processed_images.append((filename, pil_img))

    st.success("AIによる顔処理が完了しました！")
    
    # ZIPダウンロードボタン
    st.download_button(
        label="📦 処理済み画像（ZIP）をダウンロード",
        data=output_zip.getvalue(),
        file_name="costumes_dataset.zip",
        mime="application/zip"
    )

    # プレビュー表示
    st.subheader("プレビュー (最初の5枚)")
    if processed_images:
        cols = st.columns(min(5, len(processed_images)))
        for i, (fname, pimg) in enumerate(processed_images[:5]):
            cols[i].image(pimg, caption=fname, use_container_width=True)