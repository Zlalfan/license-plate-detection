"""
Pipeline inti: deteksi kendaraan+plat -> crop tiap bbox kelas 'plat' -> OCR karakter
-> gabungkan hasil -> gambar anotasi ke frame.

Ini dipakai baik untuk mode gambar tunggal maupun tiap frame video.
"""
import cv2
import numpy as np

from config import CLASS_COLORS, PLATE_REGION_CODES
from detector import detect
from plate_ocr import read_plate


def _plate_region_code(plate_text: str):
    """
    Cari kode wilayah dari huruf di DEPAN nomor plat (mis. 'EA1847M' -> ambil
    'EA'), berdasar kode wilayah regident kendaraan bermotor (Perpol No. 7
    Tahun 2021). Coba cocokkan 2 huruf dulu (kebanyakan kode di luar Jawa
    pakai 2 huruf), baru fallback ke 1 huruf (kebanyakan Jawa) kalau
    kombinasi 2 huruf itu ga ada di daftar.

    Return (kode, label) -- dua-duanya None kalau kode huruf depannya ga
    dikenali (termasuk kalau OCR kepotong sampai ga ada huruf sama sekali
    di depan).
    """
    if not plate_text:
        return None, None

    prefix = ""
    for ch in plate_text.strip().upper():
        if ch.isalpha():
            prefix += ch
        else:
            break
    if not prefix:
        return None, None

    if len(prefix) >= 2 and prefix[:2] in PLATE_REGION_CODES:
        code = prefix[:2]
        return code, PLATE_REGION_CODES[code]
    if prefix[:1] in PLATE_REGION_CODES:
        code = prefix[:1]
        return code, PLATE_REGION_CODES[code]
    return None, None


def plate_region(plate_text: str):
    """Tebak label wilayah asal plat (tanpa kode hurufnya). Return None kalau ga dikenali."""
    _, label = _plate_region_code(plate_text)
    return label


def plate_region_display(plate_text: str):
    """
    Format tampilan gabungan kode wilayah + label daerahnya, dipakai buat
    kolom 'asal plat' -- mis. 'EA (NTB — Pulau Sumbawa (Bima, Dompu, Sumbawa))'.
    Return None kalau kode huruf depannya ga dikenali.
    """
    code, label = _plate_region_code(plate_text)
    if not code:
        return None
    return f"{code} ({label})"


def _crop(image, bbox, pad_x_ratio: float = 0.18, pad_y_ratio: float = 0.08):
    """
    pad_x_ratio dibikin LEBIH BESAR dari pad_y_ratio secara sengaja: bbox
    deteksi kelas 'plat' kadang mepet banget di kiri/kanan, jadi karakter
    paling pinggir (terutama huruf terakhir) bisa kepotong sebagian dan
    bikin model karakter salah baca ujungnya. Padding vertikal sengaja
    dibiarin lebih kecil supaya ga ikut nyeret baris kecil kedua (kode
    wilayah/masa berlaku) yang ada di bawah plat -- itu sudah ditangani
    terpisah lewat _filter_main_row() di plate_ocr.py.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad_x_ratio))
    y1 = max(0, int(y1 - bh * pad_y_ratio))
    x2 = min(w, int(x2 + bw * pad_x_ratio))
    y2 = min(h, int(y2 + bh * pad_y_ratio))
    return image[y1:y2, x1:x2]


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def match_vehicle_for_plate(plate_bbox, vehicle_dets: list):
    """
    Cari deteksi kendaraan (non-'plat') yang 'memiliki' sebuah bbox plat --
    dipakai supaya tiap plat bisa dilabeli jenis kendaraannya (motor/mobil/
    bus/truck), bukan cuma teks platnya doang.

    Strategi: kendaraan yang bbox-nya memuat TITIK TENGAH bbox plat dianggap
    pemiliknya (plat fisiknya nempel di badan kendaraan, jadi titik tengah
    plat hampir pasti ada di dalam bbox kendaraan itu). Kalau ada beberapa
    kandidat yang sama-sama memuat (jarang -- biasanya karena 2 kendaraan
    saling tumpang tindih di frame), ambil yang bbox-nya paling kecil, karena
    kendaraan yang lebih jauh/di belakang cenderung bbox-nya lebih besar dan
    'menutupi' area yang lebih luas secara kebetulan.

    Kalau tidak ada kendaraan yang memuat titik tengahnya (mis. plat sedikit
    kepotong di pinggir bbox kendaraan saat deteksi kurang presisi), fallback
    ke kendaraan dengan overlap (IoU) tertinggi terhadap bbox plat.
    """
    if not vehicle_dets:
        return None

    cx, cy = _bbox_center(plate_bbox)
    containing = [
        v for v in vehicle_dets
        if v["bbox"][0] <= cx <= v["bbox"][2] and v["bbox"][1] <= cy <= v["bbox"][3]
    ]
    if containing:
        return min(containing, key=lambda v: _bbox_area(v["bbox"]))

    best = max(vehicle_dets, key=lambda v: _bbox_iou(plate_bbox, v["bbox"]))
    return best if _bbox_iou(plate_bbox, best["bbox"]) > 0 else None


def process_frame(image: np.ndarray, model_id: str, conf: float = 0.35, read_plates: bool = True):
    """
    Jalankan deteksi 5 kelas pada satu frame, lalu untuk tiap deteksi kelas
    'plat', crop dan kirim ke model OCR karakter. Tiap plat juga dipasangkan
    ke kendaraan (non-'plat') yang memuatnya, supaya hasil akhir bisa
    ditampilkan sebagai ID + jenis kendaraan + teks plat.

    Return: (annotated_image, detections)
    detections: list of dict, deteksi kelas 'plat' punya field tambahan
    'plate_text', 'vehicle_id', dan 'vehicle_class' (dua yang terakhir kosong
    kalau tidak ada kendaraan yang cocok dipasangkan).
    """
    detections = detect(model_id, image, conf=conf)

    vehicle_dets = [d for d in detections if d["class_name"] != "plat"]
    for idx, v in enumerate(vehicle_dets, start=1):
        v["vehicle_id"] = idx

    if read_plates:
        for det in detections:
            if det["class_name"] == "plat":
                crop = _crop(image, det["bbox"])
                ocr_result = read_plate(crop)
                det["plate_text"] = ocr_result["text"]
                det["plate_ocr_confidence"] = ocr_result["avg_confidence"]
                det["plate_region"] = plate_region(ocr_result["text"])

                owner = match_vehicle_for_plate(det["bbox"], vehicle_dets)
                if owner is not None:
                    det["vehicle_id"] = owner["vehicle_id"]
                    det["vehicle_class"] = owner["class_name"]

    annotated = draw_detections(image, detections)
    return annotated, detections


def _luminance(bgr):
    b, g, r = bgr
    return 0.114 * b + 0.587 * g + 0.299 * r


def _text_color_for(bg_bgr):
    return (25, 25, 25) if _luminance(bg_bgr) > 150 else (245, 245, 245)


def _rounded_rect(img, top_left, bottom_right, color, radius=8):
    """Persegi panjang solid dengan sudut membulat, dipakai buat background label."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(img, (cx, cy), radius, color, -1)


def _corner_box(img, x1, y1, x2, y2, color, thickness=3):
    """Bounding box gaya 'HUD' (sudut siku-siku) + garis tipis transparan buat konteks penuh."""
    w, h = x2 - x1, y2 - y1
    length = max(12, int(min(w, h) * 0.22))

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    for (px, py, dx, dy) in [
        (x1, y1, 1, 0), (x1, y1, 0, 1),
        (x2, y1, -1, 0), (x2, y1, 0, 1),
        (x1, y2, 1, 0), (x1, y2, 0, -1),
        (x2, y2, -1, 0), (x2, y2, 0, -1),
    ]:
        cv2.line(img, (px, py), (px + dx * length, py + dy * length), color, thickness, cv2.LINE_AA)


def _draw_label(img, x1, y1, x2, eyebrow: str, main_text: str, color):
    """
    Label dua baris (eyebrow ID+jenis, teks utama plat) dengan background
    pill warna solid sesuai kelasnya. Eyebrow (ID+jenis) dibikin lebih besar
    & agak tebal biar gampang dibaca dari jauh; teks utama (plat) tetap
    FONT_HERSHEY_SIMPLEX -- bukan font_bold (DUPLEX) yang bikin karakter
    plat mepet -- tapi thickness dinaikkan ke 3 (dari 2) biar keliatan lebih
    bold/modern tanpa ganti font, karena nebelin stroke di font yang sama
    ga bikin karakter jadi mepet kayak ganti ke DUPLEX. Sudah dicek visual:
    di scale 0.74 karakter rapat seperti "1269SD" masih kebaca jelas, ga nyatu.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    eb_scale, eb_thick = 0.52, 2
    main_scale, main_thick = 0.74, 3
    pad_x, pad_y, gap = 11, 7, 5

    (eb_w, eb_h), _ = cv2.getTextSize(eyebrow, font, eb_scale, eb_thick)
    (mw, mh), _ = cv2.getTextSize(main_text, font, main_scale, main_thick)

    box_w = max(eb_w, mw) + pad_x * 2
    box_h = eb_h + mh + gap + pad_y * 2

    place_above = (y1 - box_h) >= 0
    top = (y1 - box_h) if place_above else y1
    top = max(0, min(top, img.shape[0] - box_h))
    left = max(0, min(x1, img.shape[1] - box_w))

    _rounded_rect(img, (left, top), (left + box_w, top + box_h), color, radius=7)

    text_color = _text_color_for(color)
    eb_y = top + pad_y + eb_h
    cv2.putText(img, eyebrow, (left + pad_x, eb_y), font, eb_scale, text_color, eb_thick, cv2.LINE_AA)
    main_y = eb_y + gap + mh
    cv2.putText(img, main_text, (left + pad_x, main_y), font, main_scale, text_color, main_thick, cv2.LINE_AA)


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Label terpisah per bbox (kendaraan & plat masing-masing punya label
    sendiri). Box kendaraan nampilin ID + jenis + confidence DETEKSI
    kendaraan; box plat nampilin confidence DETEKSI plat + confidence HASIL
    BACA karakter (dua angka beda: bbox-nya yakin ga yakin vs OCR
    karakternya yakin ga yakin).

    CATATAN PENTING: pemisah dipakai " - " (strip biasa), BUKAN karakter
    "·" (middle dot). Font Hershey bawaan OpenCV ga punya glyph buat "·"
    atau "…" -- keduanya bakal kegambar sebagai "??" di frame. Ini nyebabin
    bug "MOBIL ?? 93%" yang sempet muncul; jangan pakai karakter non-ASCII
    apa pun di teks yang bakal di-cv2.putText().
    """
    annotated = image.copy()

    # gambar box kendaraan dulu, box plat belakangan biar label plat ga ketutup
    ordered = sorted(detections, key=lambda d: d["class_name"] == "plat")

    for det in ordered:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = CLASS_COLORS.get(det["class_name"], (200, 200, 200))
        _corner_box(annotated, x1, y1, x2, y2, color)

        if det["class_name"] == "plat":
            det_conf_pct = round(det["confidence"] * 100)
            plate_text = det.get("plate_text")
            if plate_text:
                ocr_conf_pct = round(det.get("plate_ocr_confidence", 0) * 100)
                eyebrow = f"PLAT {det_conf_pct}% - BACA {ocr_conf_pct}%"
                main_text = plate_text
            else:
                eyebrow = f"PLAT {det_conf_pct}%"
                main_text = "membaca..."
        else:
            eyebrow = f"{det['class_name'].upper()} {det['confidence']*100:.0f}%"
            extras = []
            vehicle_ref = det.get("track_id", det.get("vehicle_id"))
            if vehicle_ref is not None:
                extras.append(f"ID {vehicle_ref}")
            if det.get("speed_kmh") is not None:
                extras.append(f"{det['speed_kmh']:.0f} km/h")
            main_text = " - ".join(extras) if extras else det["class_name"].capitalize()

        _draw_label(annotated, x1, y1, x2, eyebrow, main_text, color)

    return annotated


def build_vehicle_records(detections: list) -> list:
    """
    Susun hasil deteksi jadi 'tabel database' satu baris per kendaraan:
    id, jenis_kendaraan, plat_nomor, asal_plat, confidence. Dipakai untuk
    mode gambar tunggal (lewat process_frame -> detections yang sudah
    dipasangkan plat-kendaraannya di match_vehicle_for_plate).

    Field & nama-nama ini SENGAJA sama persis dengan vehicles_summary di
    video_jobs.py, supaya frontend bisa pakai satu fungsi render buat
    kedua mode (gambar & video) tanpa perlu mapping field yang beda-beda.

    Kendaraan yang platnya ga kebaca / ga kedeteksi tetap dimasukkan dengan
    plat_nomor & asal_plat = None, supaya jumlah baris konsisten dengan
    jumlah kendaraan yang kedeteksi.
    """
    vehicles = {d["vehicle_id"]: d for d in detections if d["class_name"] != "plat"}

    plate_by_vehicle = {}
    for d in detections:
        if d["class_name"] == "plat" and d.get("vehicle_id") is not None:
            plate_by_vehicle[d["vehicle_id"]] = d

    records = []
    for vid in sorted(vehicles):
        vehicle = vehicles[vid]
        plate = plate_by_vehicle.get(vid)
        plate_text = plate.get("plate_text") if plate else None
        records.append({
            "id": vid,
            "jenis_kendaraan": vehicle["class_name"],
            "plat_nomor": plate_text or None,
            "asal_plat": plate_region_display(plate_text) if plate_text else None,
            "confidence": plate.get("plate_ocr_confidence") if plate else None,
        })
    return records


def summarize_counts(all_detections: list) -> dict:
    """Hitung jumlah deteksi per kelas dari list deteksi (dipakai untuk gambar tunggal)."""
    counts = {"plat": 0, "motor": 0, "mobil": 0, "bus": 0, "truck": 0}
    for det in all_detections:
        if det["class_name"] in counts:
            counts[det["class_name"]] += 1
    return counts