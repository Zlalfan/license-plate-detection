"""
Konfigurasi pusat untuk VehiScan.

Taruh file bobot (.pt) hasil training lo di folder weights/
dengan nama sesuai VEHICLE_MODELS / PLATE_OCR_MODEL di bawah.
Ganti nama file di sini kalau nama file bobot lo beda.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for d in (WEIGHTS_DIR, UPLOAD_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# 5 kelas hasil training model deteksi kendaraan + plat.
# URUTAN INI HARUS SAMA PERSIS dengan urutan class saat training (data.yaml).
VEHICLE_CLASS_NAMES = ["bus", "mobil", "motor", "plat", "truck"]

VEHICLE_MODELS = {
    "yolov5s": {
        "label": "YOLOv5s",
        "description": "Model tercepat, cocok untuk perangkat terbatas",
        "weight_file": "yolov5s_vehicle.pt",
    },
    "yolov8s": {
        "label": "YOLOv8s",
        "description": "Keseimbangan antara kecepatan dan akurasi",
        "weight_file": "yolov8s_vehicle.pt",
    },
    "yolov11s": {
        "label": "YOLOv11s",
        "description": "Akurasi tertinggi, arsitektur terbaru",
        "weight_file": "yolov11s_vehicle.pt",
    },
}

PLATE_OCR_WEIGHT_FILE = "yolov11s_plate_ocr.pt"

DEFAULT_CONF_THRESHOLD = 0.35
PLATE_OCR_CONF_THRESHOLD = 0.25

# Set True kalau server/laptop yang jalanin VehiScan RAM-nya pas-pasan
# (mis. 4GB). Normalnya, tahap normalisasi & encoding SELALU nyoba dulu di
# RESOLUSI ASLI (biar OCR plat paling tajam), baru turun bertahap kalau OOM.
# Tapi di RAM kecil, percobaan resolusi asli itu HAMPIR PASTI bikin Windows
# thrashing ke pagefile/disk berMENIT-MENIT (bukan OOM error yang cepat
# ketauan, tapi "macet" pelan yang keliatannya kayak hang) sebelum akhirnya
# jatuh ke percobaan berikutnya -- buang-buang waktu. Kalau True, langsung
# mulai dari resolusi di bawah (bukan resolusi asli), skip semua percobaan
# yang pasti gagal itu.
#
# Dua tahap di bawah SENGAJA dipisah nilainya, karena dampaknya beda:
# - NORMALIZE: frame hasil tahap ini yang dipakai buat DETEKSI KENDARAAN
#   & BACA KARAKTER PLAT (PASS 1). Downscale kelewat agresif di sini
#   langsung bikin plat yang kecil/jauh dari kamera cuma kebagian
#   segelintir piksel -- model karakter jadi ga bisa baca sama sekali.
#   Makanya nilainya dibikin CUKUP TINGGI, prioritasnya akurasi baca plat.
# - ENCODE (video hasil akhir yang di-download/ditonton): OCR plat udah
#   KELAR duluan sebelum tahap ini jalan (lihat catatan di PASS 2,
#   video_jobs.py) -- jadi downscale di sini AMAN buat akurasi baca plat,
#   cuma ngaruh ke kehalusan visual video yang ditonton. Dan tahap inilah
#   yang paling sering kena OOM (encoder libx264 lagi di-init), makanya
#   nilainya dibikin lebih KECIL/aman.
LOW_RAM_MODE = True
LOW_RAM_NORMALIZE_MAX_DIMENSION = 1600
LOW_RAM_ENCODE_MAX_DIMENSION = 960

# Berapa video yang boleh diproses BERSAMAAN (tiap job jalan di thread
# sendiri, lihat video_jobs.create_job). Dibatasi ke 1 SENGAJA: tiap job
# video sendirian aja udah lumayan berat di RAM (model deteksi + ffmpeg
# encode/decode + list deteksi per-frame buat seluruh video), jadi kalau
# dibiarin jalan bareng-bareng, gampang bikin server kehabisan memori
# (OOM) walau video-nya masing-masing sebenarnya "muat" kalau diproses
# gantian. Naikkan nilai ini cuma kalau server-nya emang punya RAM lega.
MAX_CONCURRENT_VIDEO_JOBS = 1

# Video dari HP (terutama iPhone, HEVC 4K) bisa bikin ffmpeg/libx264 kehabisan
# memori saat normalisasi orientasi & encoding (malloc gagal -> "Conversion failed!").
# Sisi terpanjang video di-downscale ke nilai ini dulu sebelum diproses, karena
# untuk kebutuhan deteksi resolusi setinggi itu ga perlu & cuma bikin server berat.
MAX_VIDEO_DIMENSION = 1920

# Selama kendaraan (track_id) masih terlihat di video, tiap kelas 'plat' terus
# dicoba dibaca ulang SAMPAI hasilnya cukup yakin (>= PLATE_LOCK_CONFIDENCE) DAN
# jumlah karakternya wajar (>= PLATE_MIN_CHARS). Ini penting karena frame-frame
# awal saat kendaraan masih bergerak/blur sering ngasih confidence lumayan
# tinggi tapi teksnya salah/kepotong -- kalau langsung dikunci di situ, hasil
# akhirnya jadi jelek walau nanti ada frame lebih jelas saat kendaraan lebih
# dekat/berhenti.
PLATE_LOCK_CONFIDENCE = 0.85
PLATE_MIN_CHARS = 5

# Skor ketajaman (Laplacian variance) dipakai buat bedain frame video yang
# masih tajam vs yang kena motion blur, karena confidence model karakter
# ga selalu turun walau frame-nya blur (bisa tetep "pede" salah baca).
# PLATE_SHARPNESS_REF: skor yang dianggap "sudah cukup tajam" (dipakai buat
#   menormalkan skor jadi 0-1 saat membandingkan 2 bacaan yang jumlah
#   karakter & confidence-nya mirip).
# PLATE_MIN_SHARPNESS_TO_ATTEMPT: kalau sudah ada bacaan yang layak tersimpan
#   DAN frame sekarang jauh lebih blur dari ini, skip percobaan OCR di frame
#   itu sama sekali (hemat kompute, dan menghindari resiko ke-overwrite oleh
#   bacaan blur yang kebetulan "pede").
PLATE_SHARPNESS_REF = 150.0
PLATE_MIN_SHARPNESS_TO_ATTEMPT = 40.0

CLASS_COLORS = {
    "plat": (0, 210, 255),
    "motor": (255, 160, 0),
    "mobil": (60, 200, 90),
    "bus": (60, 130, 246),
    "truck": (80, 80, 235),
}

# Kode wilayah huruf depan plat nomor Indonesia (Regident Ranmor sesuai Perpol
# No. 7 Tahun 2021) -> label daerah asal singkat. Coba cocokkan 2 huruf dulu
# (kebanyakan kode di luar Jawa pakai 2 huruf), baru fallback ke 1 huruf
# (kebanyakan Jawa) kalau kombinasi 2 huruf itu ga ada di daftar -- logika
# pencocokannya ada di pipeline.plate_region().
PLATE_REGION_CODES = {
    "A": "Banten (Serang, Cilegon, Pandeglang, Lebak, Kab. Tangerang)",
    "B": "DKI Jakarta & sekitarnya (Jakarta, Depok, Bekasi, Kota Tangerang, Tangsel)",
    "D": "Jawa Barat (Bandung Raya)",
    "E": "Jawa Barat (Cirebon, Indramayu, Majalengka, Kuningan)",
    "F": "Jawa Barat (Bogor, Cianjur, Sukabumi)",
    "G": "Jawa Tengah (Pekalongan, Tegal, Brebes, Batang, Pemalang)",
    "H": "Jawa Tengah (Semarang, Salatiga, Kendal, Demak)",
    "K": "Jawa Tengah (Pati, Kudus, Jepara, Rembang, Blora, Grobogan)",
    "L": "Jawa Timur (Kota Surabaya)",
    "M": "Jawa Timur (Madura: Pamekasan, Bangkalan, Sampang, Sumenep)",
    "N": "Jawa Timur (Malang Raya, Probolinggo, Pasuruan, Lumajang)",
    "P": "Jawa Timur (Besuki: Situbondo, Bondowoso, Jember, Banyuwangi)",
    "R": "Jawa Tengah (Banyumas Raya: Cilacap, Purbalingga, Banjarnegara)",
    "S": "Jawa Timur (Bojonegoro, Tuban, Lamongan)",
    "T": "Jawa Barat (Purwakarta, Karawang, Subang)",
    "W": "Jawa Timur (Gresik, Sidoarjo, Mojokerto, Jombang)",
    "Z": "Jawa Barat (Priangan Timur: Garut, Sumedang, Tasikmalaya, Ciamis, Banjar, Pangandaran)",
    "AA": "Jawa Tengah (Kedu: Magelang, Purworejo, Kebumen, Temanggung, Wonosobo)",
    "AB": "DI Yogyakarta (Yogyakarta, Bantul, Gunung Kidul, Sleman, Kulon Progo)",
    "AD": "Jawa Tengah (Solo Raya: Surakarta, Sukoharjo, Boyolali, Sragen, Karanganyar, Wonogiri, Klaten)",
    "AE": "Jawa Timur (Madiun Raya: Ngawi, Magetan, Ponorogo, Pacitan)",
    "AG": "Jawa Timur (Kediri Raya: Blitar, Tulungagung, Nganjuk, Trenggalek)",
    "BA": "Sumatera Barat (Padang & sekitarnya)",
    "BB": "Sumatera Utara bagian barat (Tapanuli, Nias)",
    "BD": "Bengkulu",
    "BE": "Lampung",
    "BG": "Sumatera Selatan (Palembang & sekitarnya)",
    "BH": "Jambi",
    "BK": "Sumatera Utara bagian timur (Medan & sekitarnya)",
    "BL": "Aceh",
    "BM": "Riau (Pekanbaru & sekitarnya)",
    "BN": "Kepulauan Bangka Belitung",
    "BP": "Kepulauan Riau (Batam & sekitarnya)",
    "DA": "Kalimantan Selatan (Banjarmasin & sekitarnya)",
    "DB": "Sulawesi Utara bagian selatan (Manado & sekitarnya)",
    "DC": "Sulawesi Barat",
    "DD": "Sulawesi Selatan bagian selatan (Makassar & sekitarnya)",
    "DE": "Maluku (Ambon & sekitarnya)",
    "DG": "Maluku Utara (Ternate & sekitarnya)",
    "DH": "NTT — Pulau Timor (Kupang & sekitarnya)",
    "DK": "Bali",
    "DL": "Sulawesi Utara (Kep. Sangihe Talaud)",
    "DM": "Gorontalo",
    "DN": "Sulawesi Tengah (Palu & sekitarnya)",
    "DP": "Sulawesi Selatan bagian utara (Parepare & sekitarnya)",
    "DR": "NTB — Pulau Lombok (Mataram & sekitarnya)",
    "DT": "Sulawesi Tenggara (Kendari & sekitarnya)",
    "DW": "Sulawesi Selatan bagian tengah (Bone, Wajo, Soppeng, Sinjai)",
    "EA": "NTB — Pulau Sumbawa (Bima, Dompu, Sumbawa)",
    "EB": "NTT — Pulau Flores",
    "ED": "NTT — Pulau Sumba",
    "KB": "Kalimantan Barat (Pontianak & sekitarnya)",
    "KH": "Kalimantan Tengah (Palangka Raya & sekitarnya)",
    "KT": "Kalimantan Timur (Balikpapan, Samarinda & sekitarnya)",
    "KU": "Kalimantan Utara (Tarakan & sekitarnya)",
    "PA": "Papua (Jayapura & sekitarnya)",
    "PB": "Papua Barat (Manokwari, Sorong & sekitarnya)",
}