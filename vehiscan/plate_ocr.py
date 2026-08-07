"""
Wrapper untuk model pembaca karakter plat (YOLOv11s).
Model ini mendeteksi tiap KARAKTER di dalam crop plat sebagai objek terpisah,
lalu kita urutkan bounding box dari kiri ke kanan buat rekonstruksi string plat.
"""
import os
from functools import lru_cache

import cv2
from ultralytics import YOLO

from config import WEIGHTS_DIR, PLATE_OCR_WEIGHT_FILE
from detector import ModelNotFoundError

# Tinggi minimum crop plat (piksel) sebelum masuk ke model OCR karakter.
# Plat kendaraan yang bergerak/jauh dari kamera sering ke-crop kecil, dan
# model karakter kesulitan baca detail huruf di crop yang terlalu kecil.
MIN_CROP_HEIGHT = 64


@lru_cache(maxsize=1)
def _load_ocr_model() -> YOLO:
    weight_path = os.path.join(WEIGHTS_DIR, PLATE_OCR_WEIGHT_FILE)
    if not os.path.exists(weight_path):
        raise ModelNotFoundError(
            f"File bobot OCR plat tidak ditemukan: {weight_path}. "
            f"Taruh file .pt di folder weights/."
        )
    return YOLO(weight_path)


def _prepare_crop(crop):
    """
    Upscale crop yang terlalu kecil + sharpen ringan (unsharp mask) supaya
    tepi karakter lebih tegas. Ini bukan solusi sempurna buat motion blur
    berat, tapi cukup membantu kasus umum: plat kecil karena jauh, atau
    sedikit blur karena kendaraan bergerak.
    """
    h, w = crop.shape[:2]
    if h < MIN_CROP_HEIGHT and h > 0:
        scale = MIN_CROP_HEIGHT / h
        crop = cv2.resize(crop, (max(1, int(w * scale)), MIN_CROP_HEIGHT), interpolation=cv2.INTER_CUBIC)

    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(crop, 1.5, blurred, -0.5, 0)
    return sharpened


def _suppress_overlapping_chars(chars: list, overlap_thresh: float = 0.45) -> list:
    """
    NMS bawaan YOLO (lewat model.predict) itu default-nya PER-KELAS. Jadi
    kalau model ragu antara 2 karakter yang bentuknya mirip di lokasi yang
    SAMA (mis. 'M' vs 'N', 'B' vs '8', '0' vs 'O'), dua-duanya bisa lolos
    sebagai 2 box terpisah karena beda kelas -- makanya "M" kebaca jadi "MN".
    agnostic_nms=True di predict() harusnya udah nangani ini, tapi kita jaga
    dobel di sini: buang box mana pun yang overlap tinggi sama box lain yang
    confidence-nya lebih tinggi, TANPA peduli kelasnya sama atau beda.
    """
    if len(chars) <= 1:
        return chars

    chars_sorted = sorted(chars, key=lambda c: c["confidence"], reverse=True)
    kept = []
    for c in chars_sorted:
        x1, y1, x2, y2 = c["bbox"]
        area_c = max(1e-6, (x2 - x1) * (y2 - y1))
        is_duplicate = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k["bbox"]
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_k = max(1e-6, (kx2 - kx1) * (ky2 - ky1))
            overlap_ratio = inter / min(area_c, area_k)
            if overlap_ratio >= overlap_thresh:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(c)
    return kept


def _filter_main_row(chars: list) -> list:
    """
    Plat Indonesia biasanya punya baris kecil KEDUA di bawah nomor utama
    (kode wilayah huruf tunggal + masa berlaku, mis. "S 04.26"). Bounding box
    kelas 'plat' dari model deteksi mencakup SELURUH fisik plat, jadi baris
    kecil itu ikut ke-crop dan kadang tertangkap sebagian oleh model karakter
    sebagai karakter tambahan yang salah di ujung teks (mis. "EA1847M" jadi
    "EA1847MN").

    Karakter baris kecil itu jauh lebih pendek (tinggi bbox-nya) dan posisi
    vertikalnya beda dari baris utama, jadi kita saring berdasarkan itu:
    ambil karakter tertinggi sebagai acuan baris utama, lalu buang karakter
    lain yang tinggi & posisi vertikalnya jauh beda dari acuan itu.
    """
    if len(chars) <= 1:
        return chars

    heights = [c["bbox"][3] - c["bbox"][1] for c in chars]
    max_h = max(heights)

    # kandidat baris utama: tinggi minimal 65% dari karakter tertinggi yang kedeteksi
    main_candidates = [c for c, h in zip(chars, heights) if h >= 0.65 * max_h]
    if not main_candidates:
        return chars

    y_centers = [(c["bbox"][1] + c["bbox"][3]) / 2 for c in main_candidates]
    main_y = sum(y_centers) / len(y_centers)
    main_h = sum(c["bbox"][3] - c["bbox"][1] for c in main_candidates) / len(main_candidates)

    filtered = [
        c for c in chars
        if abs(((c["bbox"][1] + c["bbox"][3]) / 2) - main_y) <= main_h * 0.6
    ]
    return filtered if filtered else chars


def read_plate(plate_crop, conf: float = 0.25) -> dict:
    """
    plate_crop: numpy array (BGR) hasil crop bbox kelas 'plat' dari pipeline.py

    Return dict:
        {
          "text": "B 1234 XYZ",
          "char_count": 8,
          "avg_confidence": 0.87,
          "chars": [{"char": "B", "confidence": 0.9, "bbox": [...]}, ...]
        }
    """
    if plate_crop is None or plate_crop.size == 0:
        return {"text": "", "char_count": 0, "avg_confidence": 0.0, "chars": []}

    plate_crop = _prepare_crop(plate_crop)

    model = _load_ocr_model()
    # agnostic_nms=True: NMS lintas-kelas, biar 2 karakter mirip (M/N, B/8, O/0)
    # yang kedeteksi di lokasi nyaris sama ga dua-duanya lolos cuma karena beda kelas.
    results = model.predict(plate_crop, conf=conf, agnostic_nms=True, verbose=False)[0]
    names = results.names

    chars = []
    for box in results.boxes:
        cls_idx = int(box.cls[0])
        char = names.get(cls_idx, "?")
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        chars.append({"char": char, "confidence": round(conf_score, 4), "bbox": [x1, y1, x2, y2]})

    chars = _suppress_overlapping_chars(chars)
    chars = _filter_main_row(chars)
    chars.sort(key=lambda c: c["bbox"][0])

    text = "".join(c["char"] for c in chars)
    avg_conf = round(sum(c["confidence"] for c in chars) / len(chars), 4) if chars else 0.0

    return {
        "text": text,
        "char_count": len(chars),
        "avg_confidence": avg_conf,
        "chars": chars,
    }