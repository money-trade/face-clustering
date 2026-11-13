"""หาหน้าในรูปแล้วตัดเป็นภาพขาวดำขนาดเดียวกัน
ใช้ทั้งตอนเทรน (FaceTrain_2.py) และตอนเปิดกล้อง (FaceClustering.py) ให้เทียบกันได้"""
import cv2
import numpy as np

if not hasattr(cv2, "CascadeClassifier"):
    raise ImportError(f"OpenCV {cv2.__version__} ไม่มี CascadeClassifier (ถูกถอดออกใน OpenCV 5) "
                      "ให้ติดตั้งเวอร์ชัน 4: pip install -r requirements.txt")

FACE_SIZE = (140, 180)   # (กว้าง, สูง) ของหน้าที่ตัดแล้ว = 25,200 พิกเซล
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_face(gray, rect):
    """ตัดช่วงหน้าผากถึงคาง แก้มถึงแก้ม จากกรอบที่ Haar หาได้ แล้วปรับแสง (equalizeHist)"""
    x, y, w, h = rect
    cw = w * 0.78   # กว้าง 78% ของกรอบหน้า = แก้มถึงแก้ม
    x0, y0 = int(round(x + (w - cw) / 2)), int(round(y + h * 0.06))
    x1, y1 = x0 + int(round(cw)), y0 + int(round(cw * FACE_SIZE[1] / FACE_SIZE[0]))
    p = max(0, -x0, -y0, x1 - gray.shape[1], y1 - gray.shape[0])   # กรอบเลยขอบรูป -> เติมขอบ
    if p > 0:
        gray = cv2.copyMakeBorder(gray, p, p, p, p, cv2.BORDER_REPLICATE)
    face = cv2.resize(gray[y0 + p:y1 + p, x0 + p:x1 + p], FACE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(face)


def face_from_photo(gray):
    """รูปนักเรียน (ขนาดไหนก็ได้) -> หน้าขนาด FACE_SIZE (None ถ้าหาหน้าไม่เจอ)"""
    for _ in range(4):   # ตัดเส้นขอบสีเรียบที่ติดมากับบางรูป
        if gray[0].std() < 4: gray = gray[1:]
        if gray[-1].std() < 4: gray = gray[:-1]
        if gray[:, 0].std() < 4: gray = gray[:, 1:]
        if gray[:, -1].std() < 4: gray = gray[:, :-1]
    h, w = gray.shape
    s = 600 / max(h, w)   # ให้ทุกรูปขนาดใกล้กัน
    gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    # เติมขอบก่อนหา เพราะบางรูปตัดมาชิดหน้าจนหน้าล้นขอบ
    gray = cv2.copyMakeBorder(gray, 150, 150, 150, 150, cv2.BORDER_REPLICATE)
    faces = CASCADE.detectMultiScale(gray, 1.05, 6, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return crop_face(gray, max(faces, key=lambda r: r[2] * r[3]))   # หน้าที่ใหญ่ที่สุด
