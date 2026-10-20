"""
================================================================================
KONFIGURASI SISTEM PENGENALAN WAJAH & LIVENESS DETECTION (RASPBERRY PI 3)
================================================================================
File konfigurasi terpusat untuk mengatur parameter kamera, threshold EAR,
arsitektur model, dan pencatatan metrik skripsi.
"""

import os

# --- 1. PENGATURAN KAMERA & STREAMING ---
CAMERA_INDEX = 1             # 0 untuk webcam bawaan/USB, atau path V4L2 device
FRAME_WIDTH = 320            # Resolusi rendah 320x240 sesuai kebutuhan Raspberry Pi 3
FRAME_HEIGHT = 240
TARGET_FPS = 30
DISPLAY_SCALE = 2.0          # Faktor perbesaran tampilan GUI (320x240 -> 640x480) agar OSD mudah dibaca

# --- 2. PARAMETER LIVENESS DETECTION (EYE ASPECT RATIO - EAR) ---
# Berdasarkan formula Soukupová & Čech (2016):
# - Nilai mata melek normal : 0.25 - 0.35
# - Nilai mata terpejam     : < 0.22
EAR_THRESHOLD = 0.22         # Ambang batas bukaan mata
MIN_CLOSED_FRAMES = 2        # Minimal frame berturut-turut tertutup (filter noise 1-frame)
MAX_CLOSED_FRAMES = 6        # Maksimal frame berturut-turut tertutup (mencegah foto mata tertutup)
REQUIRED_BLINKS = 1          # Jumlah kedipan sah yang dibutuhkan untuk lolos uji liveness
LIVENESS_TIMEOUT_SEC = 8.0   # Batas waktu maksimal melakukan kedipan sebelum reset
RESET_FACE_LOST_SEC = 1.5    # Reset state jika wajah tidak terdeteksi selama waktu ini
HOLD_DECISION_SEC = 3.0      # Waktu menampilkan banner hasil keputusan (Granted/Denied)

# --- 3. PARAMETER PENGENALAN WAJAH (FACE RECOGNITION) ---
DEFAULT_NAME = "Hamzah"      # Nama default pengguna untuk pendaftaran sampel wajah
LBPH_DISTANCE_THRESHOLD = 65.0  # Jarak maksimum LBPH (makin kecil makin ketat)
MOBILEFACENET_THRESHOLD = 0.40  # Cosine distance MobileFaceNet (0.0 identik, > 0.4 beda)
DLIB_TOLERANCE = 0.50           # Euclidean distance dlib 128D

# --- 4. DIREKTORI & FILE PENYIMPANAN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Pastikan folder penting selalu tersedia
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Path file penyimpanan database
LBPH_MODEL_PATH = os.path.join(MODELS_DIR, "lbph_model.xml")
LBPH_LABELS_PATH = os.path.join(MODELS_DIR, "lbph_labels.json")
MOBILEFACENET_MODEL_PATH = os.path.join(MODELS_DIR, "mobilefacenet.tflite")
MOBILEFACENET_DB_PATH = os.path.join(MODELS_DIR, "mobilefacenet_db.pkl")
DLIB_ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pickle")
DLIB_PREDICTOR_PATH = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
FACE_LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

# --- 5. LOGGING METRIK SKRIPSI ---
CSV_LOG_ENABLED = False      # Default false, dapat diaktifkan lewat parameter CLI (--log-csv)
LOG_INTERVAL_FRAMES = 1      # Frekuensi logging per-N frame
