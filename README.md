# VehiScan

Web app deteksi kendaraan (motor, mobil, bus, truck) + plat nomor, dengan pilihan
3 varian model deteksi (YOLOv5s / YOLOv8s / YOLOv11s) dan model pembaca karakter
plat terpisah (YOLOv11s).

## Struktur

```
vehiscan/
├── main.py           # FastAPI app + semua endpoint
├── config.py         # daftar model, nama kelas, warna anotasi
├── detector.py        # wrapper deteksi kendaraan+plat (5 kelas)
├── plate_ocr.py        # wrapper pembaca karakter plat
├── pipeline.py         # gabungan: deteksi -> crop plat -> OCR -> gambar anotasi
├── video_jobs.py       # job background untuk proses video + tracking + progress
├── requirements.txt
├── weights/            # >>> Model sudah ada (YOLO v5s,v8s,v11s) <<<
├── uploads/             # (otomatis) file upload sementara
├── outputs/             # (otomatis) hasil anotasi gambar/video
└── static/              # frontend
    ├── index.html
    ├── style.css
    └── script.js
```

## Cara jalanin

1. Taruh 4 file bobot lo di `weights/` (lihat `weights/README.md` buat nama
   file yang diharapkan — atau ganti nama filenya di `config.py`).

2. Install dependency (dari folder `vehiscan/`):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Jalanin server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. Buka `http://127.0.0.1:8000` di browser.

## Alur sistem

1. User pilih model deteksi (v5s/v8s/v11s) dan upload gambar atau video.
2. **Gambar**: `POST /api/detect/image` → jalan sinkron, langsung balikin hasil.
3. **Video**: `POST /api/detect/video/start` → balikin `job_id`, diproses di
   background thread pakai tracking (ByteTrack) supaya tiap kendaraan cuma
   dihitung sekali. Frontend polling `GET /api/detect/video/status/{job_id}`
   tiap 1 detik buat progress bar.
4. Tiap bbox kelas `plat` yang kedeteksi di-crop, dikirim ke model OCR
   karakter (`plate_ocr.py`), hasil karakter diurutkan kiri→kanan jadi string
   plat, lalu digambar sebagai label di frame hasil anotasi.

## Yang perlu lo sesuaikan

- **Urutan kelas** di `config.py` (`VEHICLE_CLASS_NAMES`) harus sama persis
  dengan urutan class saat training (cek `data.yaml`).
- **Nama file bobot** — default di `config.py`, ganti kalau nama file lo beda.
- **YOLOv5 dari repo asli** (bukan ultralytics package) butuh loader beda —
  ada catatan & contoh kode di bagian bawah `detector.py`.
- **Estimasi kecepatan** (yang ada di video referensi) belum diimplementasi
  di sini karena butuh kalibrasi kamera (jarak piksel↔meter). Kalau mau,
  bisa ditambahin di `video_jobs.py` pakai posisi track id antar frame +
  konstanta kalibrasi.
- Confidence threshold ada di `config.py` (`DEFAULT_CONF_THRESHOLD`,
  `PLATE_OCR_CONF_THRESHOLD`).

## Debugging

Kalau ada error, response API selalu balik JSON `{"detail": "..."}` (lihat
`unhandled_exception_handler` di `main.py`), dan traceback lengkap kecetak
di terminal tempat `uvicorn` jalan — cek situ dulu kalau ada masalah.
