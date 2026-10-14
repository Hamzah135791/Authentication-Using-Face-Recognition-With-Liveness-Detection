# ===============================================
# KONFIGURASI LIVENESS DETECTION SYSTEM
# File ini berisi semua parameter yang dapat di-tune
# ===============================================

# --------- KONFIGURASI KAMERA ---------
CAMERA_INDEX = 0                    # Index kamera (0 = default webcam)
CAMERA_WIDTH = 640                  # Lebar frame (pixel)
CAMERA_HEIGHT = 480                 # Tinggi frame (pixel)
CAMERA_FPS = 30                     # Target FPS

# --------- KONFIGURASI FACE DETECTION ---------
FACE_DETECTION_CONFIDENCE = 0.5     # Confidence threshold untuk deteksi wajah (0-1)
FACE_DATABASE_PATH = "face_database"  # Path folder database wajah

# --------- KONFIGURASI LIVENESS DETECTION (BLINK DETECTION) ---------
EAR_THRESHOLD = 0.25                # Threshold Eye Aspect Ratio untuk mendeteksi blink
# Jika EAR < threshold, dianggap mata sedang tertutup
# Nilai normal mata terbuka: ~0.40-0.50
# Nilai saat mata tertutup: ~0.10-0.15

CONSECUTIVE_FRAMES_BLINK = 2        # Jumlah frame berturut-turut dengan EAR rendah = 1 blink
# Semakin tinggi = blink lebih jelas, tapi lebih lama deteksi
# Semakin rendah = lebih sensitif, tapi risiko false positive

BLINKS_REQUIRED = 1                 # Jumlah blink yang harus terdeteksi sebelum liveness = True
# Untuk keamanan lebih, bisa naik ke 2 atau 3

EAR_CALCULATION_METHOD = "both_eyes"  # "left_eye", "right_eye", atau "both_eyes"

# --------- KONFIGURASI FACE RECOGNITION ---------
FACE_RECOGNITION_TOLERANCE = 0.6    # Tolerance untuk face matching
# Semakin kecil = lebih strict (0.3-0.4 sangat strict)
# Semakin besar = lebih permisif (0.6-0.7 permisif)
# Rekomendasi: 0.5-0.6 untuk keseimbangan

# --------- KONFIGURASI DISPLAY ---------
SHOW_EAR_VALUES = True              # Tampilkan nilai EAR di layar
SHOW_FPS = True                     # Tampilkan FPS counter
SHOW_FACIAL_LANDMARKS = True        # Tampilkan titik landmark mata (untuk debug)
FONT_THICKNESS = 2                  # Ketebalan font
FONT_SCALE = 0.7                    # Ukuran font

# --------- WARNA UNTUK DISPLAY (BGR) ---------
COLOR_GREEN = (0, 255, 0)           # Hijau: Liveness terdeteksi
COLOR_RED = (0, 0, 255)             # Merah: Access denied
COLOR_YELLOW = (0, 255, 255)        # Kuning: Blink detection / waiting
COLOR_BLUE = (255, 0, 0)            # Biru: Face detected
COLOR_WHITE = (255, 255, 255)       # Putih: Text
COLOR_BLACK = (0, 0, 0)             # Hitam: Background text

# --------- KONFIGURASI LOGGING & DEBUG ---------
DEBUG_MODE = True                   # Enable debug printing
SAVE_FAILED_FRAMES = False          # Save frame saat recognition gagal (untuk analysis)
LOG_DIRECTORY = "logs"              # Direktori untuk menyimpan logs

# --------- KONFIGURASI MEDIAPIPE ---------
MEDIAPIPE_STATIC_IMAGE_MODE = False # False untuk video (lebih cepat)
MEDIAPIPE_MAX_NUM_FACES = 1         # Jumlah wajah maksimal yang di-track
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5  # Confidence untuk deteksi
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5   # Confidence untuk tracking

# --------- INFORMASI LANDMARK MATA (untuk referensi) ---------
# MediaPipe face mesh memiliki 468 landmarks
# Index untuk mata (kiri dan kanan):
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Simplified untuk EAR calculation (6 points per eye):
# LEFT_EYE_INDICES_SIMPLE = [362, 385, 387, 263, 373, 380]
# RIGHT_EYE_INDICES_SIMPLE = [33, 160, 158, 133, 153, 145]

# --------- INFORMASI DETECTION LANDMARKS ---------
# Index untuk deteksi wajah dengan MediaPipe:
# Face bounding box menggunakan koordinat dari landmark keseluruhan
NOSE_INDEX = 1  # Ujung hidung (untuk normalisasi)
