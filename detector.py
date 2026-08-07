"""
Wrapper untuk model deteksi kendaraan + plat (5 kelas: plat, motor, mobil, bus, truck).
Mendukung 3 varian arsitektur (YOLOv5s / YOLOv8s / YOLOv11s) lewat ultralytics.YOLO.
"""
import os
import threading
from functools import lru_cache

from ultralytics import YOLO

from config import VEHICLE_CLASS_NAMES, VEHICLE_MODELS, WEIGHTS_DIR

_load_lock = threading.Lock()


class ModelNotFoundError(FileNotFoundError):
    pass


@lru_cache(maxsize=8)
def _load_model(weight_path: str) -> YOLO:
    if not os.path.exists(weight_path):
        raise ModelNotFoundError(
            f"File bobot tidak ditemukan: {weight_path}. "
            f"Taruh file .pt hasil training di folder weights/."
        )
    with _load_lock:
        model = YOLO(weight_path)
    return model


def get_vehicle_model(model_id: str) -> YOLO:
    if model_id not in VEHICLE_MODELS:
        raise ValueError(
            f"model_id '{model_id}' tidak dikenal. Pilihan: {list(VEHICLE_MODELS)}"
        )
    weight_file = VEHICLE_MODELS[model_id]["weight_file"]
    weight_path = os.path.join(WEIGHTS_DIR, weight_file)
    return _load_model(weight_path)


def detect(model_id: str, image, conf: float = 0.35):
    """
    Jalankan deteksi 5 kelas pada satu frame/gambar (numpy array BGR).
    Return: list of dict {class_name, confidence, bbox: [x1, y1, x2, y2]}
    """
    model = get_vehicle_model(model_id)
    results = model.predict(image, conf=conf, verbose=False)[0]

    detections = []
    names = results.names
    for box in results.boxes:
        cls_idx = int(box.cls[0])
        cls_name = names.get(cls_idx, str(cls_idx))
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append(
            {
                "class_name": cls_name,
                "confidence": round(conf_score, 4),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return detections


def track(model_id: str, source: str, conf: float = 0.35):
    """Deteksi + tracking (ByteTrack) pada video, buat hitung kendaraan unik & kecepatan."""
    model = get_vehicle_model(model_id)
    return model.track(
        source=source,
        conf=conf,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )


"""
Catatan soal YOLOv5:
---------------------
Ultralytics package bisa langsung load bobot YOLOv5 dengan YOLO("yolov5s_vehicle.pt")
selama file itu ditraining pakai ultralytics juga. Kalau bobot lo ditraining dari
repo asli https://github.com/ultralytics/yolov5, load-nya beda:

    import torch
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/yolov5s_vehicle.pt')

Kalau ternyata lo pakai jalur torch.hub itu, bilang aja - gue bikinin adapter terpisah.
"""