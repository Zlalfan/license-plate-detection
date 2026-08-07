import os
import shutil
import time
import traceback
import uuid

import cv2
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import VEHICLE_MODELS, UPLOAD_DIR, OUTPUT_DIR, DEFAULT_CONF_THRESHOLD
from detector import ModelNotFoundError
from pipeline import process_frame, summarize_counts, build_vehicle_records
import video_jobs

app = FastAPI(title="VehiScan API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Slider kepercayaan di UI dibatasi 50%-100%; batas ini dijaga juga di backend
# supaya request yang di-manipulasi manual (bukan lewat UI) tetap aman.
MIN_CONF_THRESHOLD = 0.5
MAX_CONF_THRESHOLD = 1.0


def _clamp_conf(value: float) -> float:
    return max(MIN_CONF_THRESHOLD, min(MAX_CONF_THRESHOLD, value))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Biar error selalu balik sebagai JSON (bukan plain text "Internal Server Error"),
    # dan tracebacknya kecetak di terminal server buat debugging.
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


@app.get("/api/models")
def list_models():
    """Daftar model deteksi yang bisa dipilih user di UI."""
    return [
        {"id": key, **info}
        for key, info in VEHICLE_MODELS.items()
    ]


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    model_id: str = Form("yolov11s"),
    conf_threshold: float = Form(DEFAULT_CONF_THRESHOLD),
):
    """Deteksi kendaraan+plat pada satu gambar, lalu baca karakter tiap plat yang kedeteksi."""
    conf_threshold = _clamp_conf(conf_threshold)

    ext = os.path.splitext(file.filename)[1] or ".jpg"
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:10]}{ext}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    image = cv2.imread(temp_path)
    if image is None:
        os.remove(temp_path)
        raise HTTPException(400, "File gambar tidak bisa dibaca")

    try:
        # Cuma bungkus process_frame() (deteksi + OCR plat) -- BUKAN termasuk
        # baca file/imwrite di luar itu, biar angkanya representasi murni
        # "kecepatan model", bukan I/O disk.
        start = time.perf_counter()
        annotated, detections = process_frame(image, model_id, conf=conf_threshold)
        elapsed = time.perf_counter() - start
        inference_fps = round(1 / elapsed, 2) if elapsed > 0 else None
    except ModelNotFoundError as e:
        raise HTTPException(503, str(e))
    finally:
        os.remove(temp_path)

    out_name = f"{uuid.uuid4().hex[:10]}_annotated.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, annotated)

    return {
        "detections": detections,
        "vehicles": build_vehicle_records(detections),
        "counts": summarize_counts(detections),
        "inference_fps": inference_fps,
        "annotated_image_url": f"/api/outputs/{out_name}",
        "conf_threshold": conf_threshold,
    }


@app.post("/api/detect/video/start")
async def start_video_job(
    file: UploadFile = File(...),
    model_id: str = Form("yolov11s"),
    conf_threshold: float = Form(DEFAULT_CONF_THRESHOLD),
):
    """Mulai pemrosesan video di background. Return job_id untuk dipolling progressnya."""
    if model_id not in VEHICLE_MODELS:
        raise HTTPException(400, f"model_id tidak dikenal: {model_id}")

    conf_threshold = _clamp_conf(conf_threshold)

    ext = os.path.splitext(file.filename)[1] or ".mp4"
    video_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:10]}{ext}")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = video_jobs.create_job(video_path, model_id, conf_threshold=conf_threshold)
    return {"job_id": job_id}


@app.get("/api/detect/video/status/{job_id}")
def video_job_status(job_id: str):
    job = video_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job tidak ditemukan")

    response = {
        "status": job["status"],
        "progress": job["progress"],
        "stage": job.get("stage"),
        "current_frame": job["current_frame"],
        "total_frames": job["total_frames"],
        "counts": job["counts"],
        "total_vehicles": job["total_vehicles"],
        "conf_threshold": job.get("conf_threshold"),
        "inference_fps": job.get("inference_fps"),
    }
    if job["status"] == "done":
        response["output_video_url"] = f"/api/outputs/{os.path.basename(job['output_video'])}"
        response["vehicles"] = job["vehicles"]
    if job["status"] == "error":
        response["error"] = job["error"]
    return response


@app.get("/api/outputs/{filename}")
def get_output_file(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File tidak ditemukan")
    return FileResponse(path)


# Serve frontend statis (index.html, style.css, script.js) di root "/"
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")