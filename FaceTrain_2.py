import os
import cv2
import numpy as np

# อ้างอิงไฟล์จากโฟลเดอร์ของสคริปต์ รันจากโฟลเดอร์ไหนก็ได้
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "ML_Data_Morning")   # โฟลเดอร์ที่เก็บรูปภาพ
MODEL_PATH = os.path.join(HERE, "knn_model.npz")

X_train, y_train = [], []

print("กำลังอ่านรูปภาพจากโฟลเดอร์...")

if os.path.exists(DATASET_DIR):
    for filename in os.listdir(DATASET_DIR):
        # รองรับไฟล์ .jpg, .jpeg, .png (ตัวพิมพ์เล็ก/ใหญ่)
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(DATASET_DIR, filename)
            x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if x is not None:
                X_train.append(x.flatten())

                # แยกชื่อเล่นหลังเครื่องหมาย _ (เช่น "6752300151_Pae.jpg" -> "Pae")
                name_part = os.path.splitext(filename)[0]
                if "_" in name_part:
                    nickname = name_part.split("_")[1].strip()
                else:
                    nickname = name_part.strip()

                y_train.append(nickname)

    if len(X_train) > 0:
        np.savez(MODEL_PATH, X=np.array(X_train), y=np.array(y_train))
        print(
            f"✅ อ่านรูปสำเร็จทั้งหมด {len(X_train)} รูป! บันทึกไฟล์ 'knn_model.npz' เรียบร้อย"
        )
    else:
        print("❌ ไม่พบไฟล์รูปภาพในโฟลเดอร์")
else:
    print(f"❌ ไม่พบโฟลเดอร์ชื่อ '{DATASET_DIR}'")