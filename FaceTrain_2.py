import os
import re
import cv2
import numpy as np

# อ้างอิงไฟล์จากโฟลเดอร์ของสคริปต์ รันจากโฟลเดอร์ไหนก็ได้
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "ML_Data_Morning")   # โฟลเดอร์ที่เก็บรูปภาพ
MODEL_PATH = os.path.join(HERE, "knn_model.npz")
IMG_EXT = (".jpg", ".jpeg", ".png")
ID_DIGITS = 10   # รหัสนักศึกษาต้องมี 10 หลัก


def is_heif(path):
    """รูปจาก iPhone (HEIF/HEIC) ที่ถูกเปลี่ยนชื่อเป็น .jpg: cv2 อ่านไม่ได้"""
    with open(path, "rb") as f:
        head = f.read(12)
    return head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"mif1", b"msf1", b"hevc")


def split_name(filename):
    """ "6752300046_Taer.jpg" -> ("6752300046", "Taer")
    รองรับตัวคั่น _ ช่องว่าง หรือ - และนามสกุลซ้อน เช่น "6752300135_Mek.jpg.jpg" """
    base = filename
    while os.path.splitext(base)[1].lower() in IMG_EXT:
        base = os.path.splitext(base)[0]
    m = re.match(r"\s*(\d+)[\s_\-]+(.+?)\s*$", base)
    return (m.group(1), m.group(2)) if m else ("", base.strip())


X_train, y_train, ids = [], [], []

print("กำลังอ่านรูปภาพจากโฟลเดอร์...")

if os.path.exists(DATASET_DIR):
    for filename in os.listdir(DATASET_DIR):
        # รองรับไฟล์ .jpg, .jpeg, .png (ตัวพิมพ์เล็ก/ใหญ่)
        if filename.lower().endswith(IMG_EXT):
            img_path = os.path.join(DATASET_DIR, filename)
            # imdecode + fromfile อ่านได้แม้ path มีภาษาไทย (cv2.imread บน Windows อ่านไม่ได้)
            x = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_GRAYSCALE)
            if x is None:
                if is_heif(img_path):
                    print(f"  ข้าม {filename}: เป็นไฟล์ HEIF (รูป iPhone) ที่ตั้งชื่อเป็น .jpg ต้องแปลงเป็น .jpg จริงก่อน")
                else:
                    print(f"  ข้าม {filename}: อ่านรูปไม่ได้")
                continue

            if x is not None:
                X_train.append(x.flatten())

                # แยกรหัสกับชื่อเล่น (เช่น "6752300151_Pae.jpg" -> "Pae")
                sid, nickname = split_name(filename)
                if not (sid.isdigit() and len(sid) == ID_DIGITS):
                    print(f"  เตือน: รหัส '{sid}' ไม่ใช่ตัวเลข {ID_DIGITS} หลัก ตรวจชื่อไฟล์ {filename}")
                ids.append(sid)
                y_train.append(nickname)

    # ชื่อเล่นซ้ำกัน (เช่น First 2 คน) ต่อท้ายด้วยเลข 4 ตัวท้ายของรหัส -> First(0607), First(1115)
    y_train = [f"{n}({s[-4:]})" if y_train.count(n) > 1 and s else n for n, s in zip(y_train, ids)]

    if len(X_train) > 0:
        np.savez(MODEL_PATH, X=np.array(X_train), y=np.array(y_train))
        print(
            f"✅ อ่านรูปสำเร็จทั้งหมด {len(X_train)} รูป! บันทึกไฟล์ 'knn_model.npz' เรียบร้อย"
        )
    else:
        print("❌ ไม่พบไฟล์รูปภาพในโฟลเดอร์")
else:
    print(f"❌ ไม่พบโฟลเดอร์ชื่อ '{DATASET_DIR}'")