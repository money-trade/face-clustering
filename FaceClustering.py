import os
import sys
import cv2
import numpy as np
from face_utils import CASCADE, crop_face

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "knn_model.npz")   # สร้างโดย FaceTrain_2.py

K = 5               # ดูเพื่อนบ้านใกล้สุดกี่คน
THRESHOLD_PCT = 10  # "คล้าย" = ใกล้กว่า 90% ของคู่เพื่อนต่างคนกันในห้อง (ยิ่งน้อยยิ่งเข้ม)


def rms_distance(X, z):
    """ระยะ Euclidean หารด้วยรากจำนวนพิกเซล = ค่าพิกเซลต่างกันเฉลี่ย (0-255) อ่านง่ายกว่าระยะดิบ"""
    return np.sqrt(np.mean((X - z) ** 2, axis=1))


def knn_similarity(X, y, z, k=K, threshold=40.0):
    """
    คำนวณระยะห่าง (Euclidean Distance) หาเพื่อนบ้านใกล้ที่สุด k คน
    threshold: ค่าระยะห่างสูงสุด ถ้ายิ่งน้อยยิ่งเหมือนกันมาก
    """
    distances = rms_distance(X, z)
    idx = np.argsort(distances)[:k]

    # เก็บผลลัพธ์คนที่ผ่านเกณฑ์ Threshold เรียงจากเหมือนมากไปน้อย
    matched_candidates = [{"name": y[i], "distance": distances[i]} for i in idx if distances[i] <= threshold]

    # ดึงชื่อคนที่ผ่านเกณฑ์แบบไม่ซ้ำกัน (คงลำดับ)
    unique_names = list(dict.fromkeys(item["name"] for item in matched_candidates))
    return unique_names, matched_candidates


def calibrate(X):
    """ตั้ง threshold จากข้อมูลจริง: ระยะระหว่างเพื่อนต่างคนกันทุกคู่ในห้อง
    คืน (threshold, ระยะกลาง) ระยะกลางใช้แปลงเป็นเปอร์เซ็นต์ความคล้าย"""
    d = np.concatenate([rms_distance(X[i + 1:], X[i]) for i in range(len(X) - 1)])
    return float(np.percentile(d, THRESHOLD_PCT)), float(np.median(d))


def recognize(gray, X, y, threshold, ref):
    """หาทุกหน้าในภาพกล้อง -> [(กรอบ, รายชื่อที่คล้าย, ระยะของคนที่ใกล้สุด, % ความคล้าย)]"""
    out = []
    for rect in CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80)):
        z = crop_face(gray, rect).flatten().astype(np.float32)   # ตัด+ปรับแสงแบบเดียวกับตอนเทรน
        names, details = knn_similarity(X, y, z, threshold=threshold)
        best = float(rms_distance(X, z).min())
        out.append((rect, names, best, max(0.0, 1 - best / ref) * 100))
    return out


def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ ไม่พบไฟล์ knn_model.npz กรุณารัน FaceTrain_2.py ก่อน")
        sys.exit(1)

    data = np.load(MODEL_PATH)
    X_train, y_train = data["X"].astype(np.float32), data["y"]
    threshold, ref = calibrate(X_train)
    print(f"โหลด {len(y_train)} คน  threshold = {threshold:.1f} (ค่าพิกเซลต่างกันเฉลี่ย)  กด q เพื่อออก")

    # CAP_DSHOW เปิดกล้องเร็วกว่าบน Windows ส่วนระบบอื่นใช้ค่าเริ่มต้น
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("เปิดกล้องไม่ได้")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = recognize(gray, X_train, y_train, threshold, ref)
        if not results:
            cv2.putText(frame, "No face found - look at the camera", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # หาหน้าเองทุกเฟรม (ไม่ต้องจัดหน้าให้ตรงกรอบ) และรองรับหลายคนพร้อมกัน
        for (x, y, w, h), names, best, sim in results:
            color = (0, 255, 0) if names else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            top = f"{names[0]} ({sim:.0f}%)" if names else f"Unknown ({sim:.0f}%)"
            cv2.putText(frame, top, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            others = ", ".join(names[1:])
            cv2.putText(frame, f"Similar to: {len(names)} person(s)" + (f" + {others}" if others else ""),
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("KNN Multi-Face Matching", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
