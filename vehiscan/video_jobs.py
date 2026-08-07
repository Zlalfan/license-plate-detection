"""
Manajer job untuk pemrosesan video secara background, supaya frontend bisa
polling progress (mirip "Processing frame 500/736" di referensi).

Pakai tracking (ByteTrack via ultralytics) supaya tiap kendaraan cuma dihitung
sekali walau muncul di banyak frame, dan supaya plat tiap kendaraan cuma perlu
dibaca sekali (begitu track id baru muncul & plat-nya kebaca dengan confidence
cukup tinggi, cache-kan hasilnya).
"""
import gc
import os
import shutil
import subprocess
import time
import uuid
import threading

import cv2

from config import (
    OUTPUT_DIR,
    VEHICLE_CLASS_NAMES,
    DEFAULT_CONF_THRESHOLD,
    MAX_VIDEO_DIMENSION,
    MAX_CONCURRENT_VIDEO_JOBS,
    LOW_RAM_MODE,
    LOW_RAM_NORMALIZE_MAX_DIMENSION,
    LOW_RAM_ENCODE_MAX_DIMENSION,
    PLATE_LOCK_CONFIDENCE,
    PLATE_MIN_CHARS,
    PLATE_SHARPNESS_REF,
    PLATE_MIN_SHARPNESS_TO_ATTEMPT,
)
from detector import get_vehicle_model
from pipeline import draw_detections, _crop, match_vehicle_for_plate, plate_region, plate_region_display
from plate_ocr import read_plate

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Batasin berapa job video yang beneran JALAN (bukan cuma "queued" nunggu)
# di saat yang sama, biar RAM ga direbutin banyak proses ffmpeg+model
# sekaligus. Job yang belum kebagian slot bakal nunggu di _run_job sebelum
# mulai kerja, statusnya tetap "queued" selama nunggu.
_video_slot = threading.Semaphore(MAX_CONCURRENT_VIDEO_JOBS)


def _get_ffmpeg_path():
    """
    Cari binary ffmpeg: pakai yang terinstall di sistem kalau ada,
    kalau ga ada fallback ke binary bawaan package imageio-ffmpeg
    (biar ga wajib install ffmpeg manual di OS).
    """
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _scale_filter_args(max_dimension: int | None) -> list:
    """
    Bangun argumen -vf buat ffmpeg. Filter dipecah 2 tahap SENGAJA supaya
    kompatibel dengan ffmpeg versi lama: tahap pertama pakai
    'force_original_aspect_ratio=decrease' (opsi lama, aman), tahap kedua
    genapin lebar/tinggi manual lewat trunc(iw/2)*2 -- BUKAN pakai opsi
    'force_divisible_by' (baru ada di ffmpeg >= 4.3/4.4). Genap wajib karena
    yuv420p butuh dimensi kelipatan 2.
    """
    if max_dimension:
        scale_filter = (
            f"scale='min({max_dimension},iw)':'min({max_dimension},ih)':"
            f"force_original_aspect_ratio=decrease,"
            f"scale='trunc(iw/2)*2':'trunc(ih/2)*2'"
        )
    else:
        scale_filter = "scale='trunc(iw/2)*2':'trunc(ih/2)*2'"
    return ["-vf", scale_filter]


def _run_ffmpeg_encode(cmd: list) -> None:
    """Jalankan command ffmpeg, lempar RuntimeError dengan tag 'OOM' kalau kegagalannya soal memori."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = result.stderr[-800:]
        if "malloc" in stderr_tail.lower() or "conversion failed" in stderr_tail.lower():
            raise RuntimeError(f"OOM saat encoding. Detail: {stderr_tail}")
        raise RuntimeError(f"ffmpeg gagal: {stderr_tail}")


def _scaled_dims(width: int, height: int, max_dimension: int | None) -> tuple:
    """
    Hitung lebar/tinggi OUTPUT setelah di-downscale supaya sisi terpanjangnya
    ga lebih dari max_dimension (None = ga di-downscale). Dibulatkan genap
    karena yuv420p butuh dimensi kelipatan 2.
    """
    if not max_dimension or max(width, height) <= max_dimension:
        w, h = width, height
    else:
        scale = max_dimension / max(width, height)
        w, h = round(width * scale), round(height * scale)
    w -= w % 2
    h -= h % 2
    return max(2, w), max(2, h)


def _render_and_encode_once(
    job_id: str,
    normalized_path: str,
    frame_detections: list,
    tracked_plates: dict,
    fps: float,
    width: int,
    height: int,
    final_path: str,
    max_dimension: int | None,
    low_memory: bool,
) -> None:
    """
    Baca ulang normalized_path frame demi frame, gambar anotasi pakai bacaan
    plat FINAL, lalu KIRIM LANGSUNG (pipe stdin) ke proses ffmpeg yang encode
    ke H.264 -- TANPA nulis file mentah (mp4v) resolusi asli dulu ke disk.

    Ini beda dari pendekatan lama (tulis raw_path lewat cv2.VideoWriter, baru
    ffmpeg transcode raw_path itu dengan filter -vf buat downscale): dengan
    cara lama, ffmpeg WAJIB DECODE dulu seluruh raw_path di RESOLUSI ASLI
    sebelum sempat nge-apply filter downscale -- jadi walau target output
    di-set kecil (960px dst), sisi DECODE-nya tetap berat & butuh memori
    besar, cuma sisi ENCODE yang beneran diperkecil. Itu sebabnya fallback
    "turunin resolusi" bisa tetap OOM sampai ke percobaan terkecil.

    Dengan pipe, frame di-resize di PYTHON (pakai cv2.resize) SEBELUM
    dikirim ke ffmpeg -- jadi ffmpeg dari awal cuma pernah "lihat" frame di
    ukuran kecil itu, ga pernah nanggung beban decode resolusi asli sama
    sekali. Ini juga otomatis ngilangin satu putaran decode+encode penuh
    (raw_path ga pernah ada), yang tadinya jadi beban memori & waktu ekstra
    sendiri.
    """
    ffmpeg = _get_ffmpeg_path()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg tidak ditemukan. Install ffmpeg (https://ffmpeg.org/download.html) "
            "atau jalankan: pip install imageio-ffmpeg"
        )

    out_w, out_h = _scaled_dims(width, height, max_dimension)
    needs_resize = (out_w, out_h) != (width, height)
    preset = "ultrafast" if low_memory else "fast"
    low_mem_args = ["-refs", "1", "-bf", "0", "-threads", "2"] if low_memory else []

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.3f}",
        "-i", "-",
        "-c:v", "libx264", "-preset", preset, "-crf", "23",
        *low_mem_args,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        final_path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    # PENTING: ffmpeg nulis progress/warning ke stderr terus-menerus selama
    # encoding. Kalau stderr itu ga ada yang "nyedot" SAMBIL kita nulis frame
    # ke stdin, pipe stderr-nya bisa penuh (buffer OS biasanya cuma ~64KB) --
    # begitu penuh, ffmpeg BERHENTI nunggu stderr-nya dibaca, yang otomatis
    # bikin dia juga berhenti nyedot stdin, yang bikin proc.stdin.write() di
    # bawah nge-BLOCK SELAMANYA (deadlock, bukan lambat/nge-hang beneran).
    # Makanya progress-nya kejadian macet konsisten di frame tertentu, bukan
    # random -- itu titik pas buffer stderr penuh. -loglevel error di atas
    # udah minimalin ini, tapi thread ini tetap dipasang sebagai jaring
    # pengaman kalau ffmpeg tetap ngeprint sesuatu (mis. warning codec).
    stderr_chunks: list[bytes] = []

    def _drain_stderr(pipe):
        for line in iter(pipe.readline, b""):
            stderr_chunks.append(line)
            if len(stderr_chunks) > 200:  # jaga-jaga biar list-nya ga membengkak
                del stderr_chunks[:100]

    stderr_thread = threading.Thread(target=_drain_stderr, args=(proc.stderr,), daemon=True)
    stderr_thread.start()

    cap2 = cv2.VideoCapture(normalized_path)
    total_draw_frames = len(frame_detections)
    draw_idx = 0
    pipe_broke = False
    try:
        while True:
            ok, frame = cap2.read()
            if not ok or draw_idx >= total_draw_frames:
                break

            detections = frame_detections[draw_idx]
            for det in detections:
                if det["class_name"] == "plat":
                    final = tracked_plates.get(det.get("vehicle_track_id"))
                    if final:
                        det["plate_text"] = final["text"]
                        det["plate_ocr_confidence"] = final["confidence"]
                        det["plate_region"] = plate_region(final["text"])

            annotated = draw_detections(frame, detections)
            if needs_resize:
                annotated = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_AREA)

            try:
                proc.stdin.write(annotated.tobytes())
            except (BrokenPipeError, OSError):
                # ffmpeg-nya mati duluan (biasanya karena OOM juga) -- stop
                # ngirim frame, biar kita bisa baca stderr-nya di bawah.
                pipe_broke = True
                break

            draw_idx += 1
            if draw_idx % 5 == 0 or draw_idx == total_draw_frames:
                progress = 65.0 + (draw_idx / max(total_draw_frames, 1)) * 28.0
                _update(
                    job_id, progress=round(progress, 1),
                    stage=f"Menggambar anotasi frame {draw_idx}/{total_draw_frames} (tahap 2/2)…",
                )
    finally:
        cap2.release()
        try:
            proc.stdin.close()
        except OSError:
            pass

    stderr_thread.join(timeout=10)
    returncode = proc.wait()
    stderr_tail = b"".join(stderr_chunks)[-800:].decode(errors="replace")

    if returncode != 0 or pipe_broke:
        if pipe_broke or "malloc" in stderr_tail.lower() or "conversion failed" in stderr_tail.lower():
            raise RuntimeError(f"OOM saat encoding. Detail: {stderr_tail}")
        raise RuntimeError(f"ffmpeg gagal: {stderr_tail}")


def _render_and_encode(
    job_id: str,
    normalized_path: str,
    frame_detections: list,
    tracked_plates: dict,
    fps: float,
    width: int,
    height: int,
    final_path: str,
) -> None:
    """
    Coba _render_and_encode_once() dari resolusi asli, turun bertahap kalau
    OOM. gc.collect() dipanggil sebelum tiap percobaan buat bantu bebasin
    memori yang masih dipegang objek Python (mis. dari percobaan
    sebelumnya) sebelum ffmpeg baru dijalankan.

    CATATAN: kalau tetap OOM bahkan di percobaan terkecil dengan malloc
    yang sangat kecil (~kilobyte), itu biasanya nandain server memang udah
    nyaris ga ada RAM bebas sama sekali saat itu -- bukan cuma soal video
    ini gede, tapi kemungkinan proses lain (atau job video lain yang lagi
    jalan bareng) juga makan RAM. Downscale sejauh apa pun ga akan nolong
    kalau sistemnya sendiri sudah kehabisan.
    """
    attempts = (
        [
            {"max_dimension": LOW_RAM_ENCODE_MAX_DIMENSION, "low_memory": True,
             "label": f"maks {LOW_RAM_ENCODE_MAX_DIMENSION}px, mode hemat memori"},
            {"max_dimension": 640, "low_memory": True, "label": "maks 640px, mode hemat memori"},
        ]
        if LOW_RAM_MODE
        else [
            {"max_dimension": None, "low_memory": False, "label": "resolusi asli"},
            {"max_dimension": None, "low_memory": True, "label": "resolusi asli, mode hemat memori"},
            {"max_dimension": MAX_VIDEO_DIMENSION, "low_memory": True, "label": f"maks {MAX_VIDEO_DIMENSION}px, mode hemat memori"},
            {"max_dimension": 1280, "low_memory": True, "label": "maks 1280px, mode hemat memori"},
            {"max_dimension": 960, "low_memory": True, "label": "maks 960px, mode hemat memori"},
            {"max_dimension": 640, "low_memory": True, "label": "maks 640px, mode hemat memori"},
        ]
    )
    last_error: Exception | None = None

    for attempt in attempts:
        try:
            gc.collect()
            _update(job_id, progress=65.0, stage=f"Menggambar & menyusun video hasil ({attempt['label']})…")
            _render_and_encode_once(
                job_id, normalized_path, frame_detections, tracked_plates,
                fps, width, height, final_path,
                max_dimension=attempt["max_dimension"], low_memory=attempt["low_memory"],
            )
            return
        except RuntimeError as e:
            last_error = e
            if "OOM" not in str(e):
                raise
            continue

    raise RuntimeError(
        "ffmpeg tetap gagal karena kehabisan memori (OOM) saat menyusun video hasil, "
        "meski sudah dicoba sampai resolusi kecil (640px) & mode hemat memori. Ini "
        "kemungkinan besar bukan lagi soal ukuran video, tapi RAM bebas di server "
        "memang sudah nyaris habis saat itu (mis. karena job lain jalan bareng, atau "
        "RAM server terlalu kecil buat beban ini). Coba tambah RAM/swap di server, "
        "atau batasi jumlah job video yang diproses bersamaan. "
        f"Detail terakhir: {last_error}"
    )


def _normalize_orientation(input_path: str, output_path: str, max_dimension: int | None = None):
    """
    Video dari HP/dashcam sering nyimpen rotasi sebagai METADATA (mis. "putar 90°"),
    bukan di piksel aslinya. OpenCV/ultralytics baca piksel MENTAH dan
    MENGABAIKAN metadata itu, jadi model 'lihat' framenya miring -> deteksi
    jelek/kosong dan video hasil anotasi ikut miring.

    ffmpeg secara default MENERAPKAN rotasi metadata pas decode, jadi re-encode
    di sini otomatis "membakar" rotasi yang benar ke piksel. Hasilnya: video
    tegak beneran, konsisten dipakai baik untuk deteksi maupun ditulis ulang.

    max_dimension: kalau None, video diproses di RESOLUSI ASLI (plat paling
    tajam buat OCR). Kalau di-isi, video di-downscale dulu supaya sisi
    terpanjangnya ga lebih dari nilai itu -- dipakai sebagai FALLBACK kalau
    encode di resolusi asli kehabisan memori (lihat _normalize_orientation_with_retry).
    Downscale itu MEMOTONG detail plat secara permanen sebelum sempat di-crop
    buat OCR, jadi cuma dipakai kalau kepepet, bukan default.
    """
    ffmpeg = _get_ffmpeg_path()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg tidak ditemukan. Install ffmpeg (https://ffmpeg.org/download.html) "
            "atau jalankan: pip install imageio-ffmpeg"
        )

    vf_args = _scale_filter_args(max_dimension)
    cmd = [
        ffmpeg, "-y", "-i", input_path,
        *vf_args,
        # CRF SANGAT rendah (nyaris lossless) SENGAJA -- file ini cuma
        # perantara sementara (dihapus setelah job selesai) yang jadi SUMBER
        # PIXEL buat deteksi & OCR karakter plat. CRF 20 sebelumnya bikin
        # ukuran file kecil, tapi kompresinya cukup ngerusak detail tepi
        # karakter kecil sampai model salah baca (mis. "M" jadi kelihatan
        # "N"/"L") -- padahal gambar/frame asli yang TANPA re-encode selalu
        # kebaca sempurna. Ukuran file gede di sini ga masalah krn temporary.
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "10",
        "-pix_fmt", "yuv420p",
        "-metadata:s:v:0", "rotate=0",  # buang tag rotasi, karena udah dibakar ke piksel
        # -an: BUANG audio sama sekali. File ini cuma sumber PIKSEL buat
        # deteksi & buat dibaca ulang di PASS 2 -- videonya sendiri ga
        # pernah punya track audio (di-pipe langsung ke encoder final tanpa
        # audio, lihat _render_and_encode_once). Sebelumnya audio ikut
        # di-decode+re-encode (aac->aac) di sini padahal ga pernah kepake
        # sama sekali -- kerja & alokasi memori sia-sia yang bisa ikut
        # nyumbang ke OOM pas encoder video mau di-init.
        "-an",
        output_path,
    ]
    _run_ffmpeg_encode(cmd)


def _normalize_orientation_with_retry(input_path: str, output_path: str, job_id: str | None = None):
    """
    Coba proses di resolusi ASLI dulu (plat paling tajam buat OCR). Kalau
    server kehabisan memori (OOM) di resolusi itu, baru turunkan resolusi
    bertahap sampai berhasil. Jadi downscale cuma kejadian kalau memang
    kepepet, bukan default buat semua video -- ini yang bikin hasil OCR
    plat sebelumnya jadi lebih buruk padahal videonya sama.
    """
    attempts = (
        [LOW_RAM_NORMALIZE_MAX_DIMENSION, 1280, 960, 640]
        if LOW_RAM_MODE
        else [None, MAX_VIDEO_DIMENSION, 1280, 960, 640]
    )
    last_error: Exception | None = None

    for max_dim in attempts:
        try:
            if job_id:
                label = "resolusi asli" if max_dim is None else f"maks {max_dim}px"
                _update(job_id, stage=f"Menormalisasi orientasi video ({label})…")
            _normalize_orientation(input_path, output_path, max_dimension=max_dim)
            return
        except RuntimeError as e:
            last_error = e
            if "OOM" not in str(e):
                # bukan soal memori (mis. file corrupt) -> ga ada gunanya diulang lebih kecil
                raise
            continue

    raise RuntimeError(
        "ffmpeg tetap gagal karena kehabisan memori (OOM) meski sudah dicoba sampai "
        f"resolusi terkecil (640px). Server ini kemungkinan RAM-nya terlalu mepet "
        f"untuk video ini saat itu. Detail terakhir: {last_error}"
    )


def create_job(video_path: str, model_id: str, conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",       # queued -> processing -> done | error
            "progress": 0.0,
            "stage": "Menunggu giliran…",
            "current_frame": 0,
            "total_frames": 0,
            "model_id": model_id,
            "conf_threshold": conf_threshold,
            "output_video": None,
            "counts": {c: 0 for c in VEHICLE_CLASS_NAMES if c != "plat"},
            "total_vehicles": 0,
            "tracked_plates": {},     # track_id -> best plate text seen so far
            "vehicles": [],           # ringkasan akhir per kendaraan unik
            "inference_fps": None,    # kecepatan proses PASS 1 (deteksi+tracking), frame video/detik
            "error": None,
        }
    threading.Thread(
        target=_run_job, args=(job_id, video_path, model_id, conf_threshold), daemon=True
    ).start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def _sharpness_score(crop) -> float:
    """
    Skor ketajaman pakai variance Laplacian: makin kabur crop-nya, makin
    kecil skornya. Dipakai buat prioritaskan frame yang tajam saat milih
    bacaan plat terbaik dari banyak frame video.
    """
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _is_better_reading(new_ocr: dict, new_sharpness: float, prev: dict | None) -> bool:
    """
    Bandingkan hasil OCR baru vs yang sudah tersimpan untuk track_id yang sama.
    Prioritas: jumlah karakter yang wajar dulu (biar ga kepilih bacaan yang
    kepotong 1-2 huruf gara-gara blur/sudut kamera). Kalau jumlah karakternya
    sama, baru dibandingkan skor gabungan confidence x ketajaman -- ini biar
    bacaan dari frame yang blur ga menang cuma gara-gara kebetulan confidence-nya
    sedikit lebih tinggi padahal frame-nya jelas lebih buram.
    """
    if prev is None:
        return True
    new_ok = new_ocr["char_count"] >= PLATE_MIN_CHARS
    prev_ok = prev["char_count"] >= PLATE_MIN_CHARS
    if new_ok and not prev_ok:
        return True
    if prev_ok and not new_ok:
        return False
    if new_ocr["char_count"] != prev["char_count"]:
        return new_ocr["char_count"] > prev["char_count"]

    new_sharpness_factor = min(1.0, new_sharpness / PLATE_SHARPNESS_REF)
    prev_sharpness_factor = min(1.0, prev.get("sharpness", PLATE_SHARPNESS_REF) / PLATE_SHARPNESS_REF)
    new_score = new_ocr["avg_confidence"] * new_sharpness_factor
    prev_score = prev["confidence"] * prev_sharpness_factor
    return new_score > prev_score


def _run_job(job_id: str, video_path: str, model_id: str, conf_threshold: float):
    normalized_path = None
    _update(job_id, stage="Menunggu giliran diproses (server cuma proses "
                          f"{MAX_CONCURRENT_VIDEO_JOBS} video sekaligus biar RAM ga rebutan)…")
    _video_slot.acquire()
    try:
        # Tahap 1: load model (bisa lambat kalau ini load pertama kali / CPU-only).
        _update(job_id, status="processing", progress=1.0, stage="Memuat model deteksi…")
        model = get_vehicle_model(model_id)

        # Tahap 2: normalisasi orientasi (+ downscale kalau kepepet OOM) -> semua proses berikutnya pakai video ini
        _update(job_id, progress=4.0, stage="Menormalisasi orientasi video…")
        normalized_path = os.path.join(OUTPUT_DIR, f"{job_id}_normalized.mp4")
        _normalize_orientation_with_retry(video_path, normalized_path, job_id=job_id)

        cap = cv2.VideoCapture(normalized_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        seen_track_ids = {c: set() for c in VEHICLE_CLASS_NAMES if c != "plat"}
        tracked_plates: dict[int, dict] = {}
        # Simpen detection PER FRAME (bukan pixel-nya, cuma metadata kecil: kelas,
        # confidence, bbox, track_id) -- dipakai lagi di PASS 2 buat gambar ulang.
        # Ini SENGAJA dipisah dari nulis video supaya bacaan plat yang ditampilkan
        # bisa konsisten pakai hasil AKHIR/TERBAIK dari seluruh video, bukan
        # snapshot "sejauh video diproses sampai frame itu" seperti sebelumnya
        # (itu yang bikin plat sama kadang kebaca benar kadang salah tergantung
        # di frame mana dia pertama/terakhir "dikunci").
        frame_detections: list[list[dict]] = []
        frame_idx = 0

        # ---- PASS 1: deteksi + tracking + kumpulin bacaan plat TERBAIK per kendaraan ----
        _update(job_id, progress=8.0, stage="Menganalisis frame (tahap 1/2)…")
        stream = model.track(
            source=normalized_path, conf=conf_threshold, stream=True,
            persist=True, tracker="bytetrack.yaml", verbose=False,
        )

        # inference_fps dihitung dari PASS 1 doang (deteksi+tracking+OCR
        # plat) -- bukan dari PASS 2 (gambar anotasi + encode) yang beban
        # kerjanya beda banget (ffmpeg encode, dll). Ini yang paling
        # merepresentasikan "kecepatan model" beneran.
        pass1_start = time.perf_counter()

        for result in stream:
            frame_idx += 1
            frame = result.orig_img
            names = result.names
            detections = []

            boxes = result.boxes
            has_ids = boxes.id is not None
            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i])
                cls_name = names.get(cls_idx, str(cls_idx))
                conf_score = float(boxes.conf[i])
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i]]
                track_id = int(boxes.id[i]) if has_ids else None

                det = {
                    "class_name": cls_name, "confidence": round(conf_score, 4),
                    "bbox": [x1, y1, x2, y2], "track_id": track_id,
                }

                if cls_name in seen_track_ids and track_id is not None:
                    seen_track_ids[cls_name].add(track_id)

                detections.append(det)

            # ---- pasangkan tiap deteksi 'plat' di frame ini ke kendaraan yang
            # memuatnya (bukan ke track_id plat itu sendiri), lalu simpan bacaan
            # OCR terbaik per KENDARAAN (track_id kendaraan) -- ini yang bikin
            # hasil akhir bisa dilabeli jenis kendaraannya (motor/mobil/dst),
            # bukan cuma teks platnya doang.
            vehicle_dets_in_frame = [
                d for d in detections if d["class_name"] != "plat" and d["track_id"] is not None
            ]
            for det in detections:
                if det["class_name"] != "plat":
                    continue

                owner = match_vehicle_for_plate(det["bbox"], vehicle_dets_in_frame)
                if owner is None:
                    # Ga ada kendaraan yang cocok dipasangkan di frame ini
                    # (jarang -- mis. kendaraan induknya kebetulan ga kedeteksi
                    # di frame yang sama) -> plat ini dilewati dulu.
                    continue

                vehicle_track_id = owner["track_id"]
                det["vehicle_track_id"] = vehicle_track_id
                det["vehicle_class"] = owner["class_name"]

                prev = tracked_plates.get(vehicle_track_id)
                prev_locked = (
                    prev is not None
                    and prev["confidence"] >= PLATE_LOCK_CONFIDENCE
                    and prev["char_count"] >= PLATE_MIN_CHARS
                )
                if prev_locked:
                    continue

                crop = _crop(frame, det["bbox"])
                sharpness = _sharpness_score(crop)

                # Kalau sudah ada bacaan yang layak DAN frame sekarang jauh
                # lebih blur, skip aja -- hemat kompute & hindari resiko
                # ke-overwrite bacaan bagus oleh bacaan dari frame buram.
                prev_decent = prev is not None and prev["char_count"] >= PLATE_MIN_CHARS
                skip_blurry_frame = prev_decent and sharpness < PLATE_MIN_SHARPNESS_TO_ATTEMPT

                if skip_blurry_frame:
                    continue

                ocr = read_plate(crop)
                if ocr["text"] and _is_better_reading(ocr, sharpness, prev):
                    tracked_plates[vehicle_track_id] = {
                        "text": ocr["text"],
                        "confidence": ocr["avg_confidence"],
                        "char_count": ocr["char_count"],
                        "class_name": owner["class_name"],
                        "sharpness": sharpness,
                    }
                # CATATAN: sengaja TIDAK isi det["plate_text"] di sini. Itu baru
                # diisi di PASS 2 pakai bacaan FINAL, biar semua frame konsisten.

            frame_detections.append(detections)

            # Update tiap frame supaya progress kelihatan jalan terus dari awal.
            counts = {c: len(ids) for c, ids in seen_track_ids.items()}
            frame_fraction = frame_idx / max(total_frames, frame_idx, 1)
            elapsed = time.perf_counter() - pass1_start
            inference_fps = round(frame_idx / elapsed, 2) if elapsed > 0 else None
            _update(
                job_id,
                current_frame=frame_idx,
                total_frames=total_frames or frame_idx,
                progress=round(8.0 + frame_fraction * 55.0, 1),
                stage=f"Menganalisis frame {frame_idx}/{total_frames or '?'} (tahap 1/2)…",
                counts=counts,
                total_vehicles=sum(counts.values()),
                inference_fps=inference_fps,
            )

        # ---- PASS 2: baca ulang video, gambar anotasi pakai bacaan plat FINAL,
        # langsung di-pipe ke ffmpeg buat di-encode (lihat _render_and_encode) ----
        # Sekarang tracked_plates sudah berisi bacaan TERBAIK dari SELURUH video
        # untuk tiap track_id, jadi satu kendaraan bakal tampil dengan teks plat
        # yang SAMA di semua frame -- bukan berubah-ubah tergantung kapan dia
        # "ketemu" bacaan itu.
        final_path = os.path.join(OUTPUT_DIR, f"{job_id}_annotated.mp4")
        _render_and_encode(
            job_id, normalized_path, frame_detections, tracked_plates,
            fps, width, height, final_path,
        )

        # Nama kolom SENGAJA disamakan dengan build_vehicle_records() di
        # pipeline.py (dipakai mode gambar tunggal), supaya frontend bisa
        # render "tabel database" hasil deteksi dengan struktur yang sama
        # buat gambar maupun video.
        vehicles_summary = [
            {
                "id": tid,
                "jenis_kendaraan": info["class_name"],
                "plat_nomor": info["text"],
                "asal_plat": plate_region_display(info["text"]),
                "confidence": info["confidence"],
            }
            for tid, info in tracked_plates.items()
        ]

        _update(
            job_id, status="done", progress=100.0, stage="Selesai",
            output_video=final_path, vehicles=vehicles_summary,
        )
    except Exception as e:
        _update(job_id, status="error", stage="Gagal", error=str(e))
    finally:
        if normalized_path and os.path.exists(normalized_path):
            os.remove(normalized_path)
        _video_slot.release()


def _update(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)