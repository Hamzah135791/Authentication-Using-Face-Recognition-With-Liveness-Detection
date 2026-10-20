"""
================================================================================
SISTEM AUTENTIKASI PENGENALAN WAJAH DENGAN DETEKSI KEASLIAN (LIVENESS DETECTION)
================================================================================
Program ini dirancang untuk kebutuhan Tugas Akhir / Skripsi dengan fokus pada:
1. Pengenalan Wajah (Face Recognition) menggunakan Deep Metric Learning (128-D).
2. Deteksi Kehidupan (Liveness Detection) berbasis kedipan mata menggunakan
   Eye Aspect Ratio (EAR) dari 68 Facial Landmarks.
3. Optimasi Real-Time (Frame Skipping & Adaptive Tracking) agar tidak lag.

Alur Kerja Sistem (Flow):
[Kamera] -> [Deteksi Wajah dlib HOG] -> [Ekstraksi 68 Titik Landmark Mata]
   |                                               |
   v                                               v
[Pencocokan Identitas Wajah]            [Hitung Rasio Bukaan Mata (EAR)]
   |                                               |
   +-----------------------+-----------------------+
                           |
                           v
              [Pengambilan Keputusan Status]:
   - Tidak ada wajah       -> Pandu pengguna ke depan kamera
   - Wajah asing           -> Akses Ditolak (Unknown)
   - Wajah cocok, blm kedip -> Pandu pengguna untuk berkedip (Uji Liveness)
   - Wajah cocok & kedip   -> Akses Diterima (Verified Real Person)
================================================================================
"""

import os
import pickle
import time
import cv2
import numpy as np
import face_recognition


# ==============================================================================
# 1. PENGATURAN DAN PARAMETER SISTEM (CONFIGURATION)
# ==============================================================================

# --- A. Data Pengguna ---
NAMA_PENGGUNA = "Hamzah"           # Nama yang akan didaftarkan saat menekan tombol 'e'
FILE_DATABASE = "encodings.pickle" # File penyimpanan pola fitur wajah

# --- B. Kamera ---
CAM_INDEX = 1        # Indeks kamera (coba 0 untuk webcam internal, 1/2 untuk eksternal)
FRAME_WIDTH = 640    # Lebar resolusi kamera (640x480 adalah standar optimal & ringan)
FRAME_HEIGHT = 480   # Tinggi resolusi kamera

# --- C. Parameter Liveness Detection (Kedipan Mata / EAR) ---
# EAR (Eye Aspect Ratio): Rasio antara tinggi bukaan mata dan lebar sudut mata.
# Saat mata melek normal, nilainya sekitar 0.25 - 0.35.
# Saat mata terpejam, nilainya drop di bawah 0.22.
EAR_THRESHOLD = 0.22         # Batas nilai EAR: Di bawah ini dianggap "mata terpejam"
MIN_CLOSED_FRAMES = 1        # Minimal frame mata terpejam (1-2 frame cukup jika FPS tinggi)
MAX_CLOSED_FRAMES = 8        # Maksimal frame terpejam (mencegah orang memejamkan mata terus/tidur)
BLINK_VALID_SEC = 6.0        # Masa berlaku kedipan (dalam detik) sebelum kadaluarsa
REQUIRED_BLINKS = 1          # Jumlah kedipan yang dibutuhkan untuk lolos liveness

# --- D. Parameter Pengenalan Wajah (Face Recognition) ---
# Tolerance: Jarak Euclidean antara wajah kamera dan database (makin kecil makin ketat).
# 0.60 = default dlib, 0.50 = cukup ketat & aman, 0.45 = sangat ketat.
TOLERANCE = 0.50
RECOG_INTERVAL_SEC = 0.6     # Interval pengenalan wajah (tiap 0.6 detik agar hemat CPU)

# --- E. Parameter Optimasi Kinerja & FPS (Performance Tuning) ---
# FRAME SKIPPING: Mencari posisi wajah membutuhkan komputasi berat (HOG).
# Dengan menjalankan pencarian posisi wajah tiap N frame, FPS meningkat dari ~10 ke ~25-30 FPS.
PROCESS_EVERY_N_FRAMES = 2   # Jalankan deteksi lokasi wajah tiap 2 frame
DETECT_SCALE = 0.5           # Gambar diperkecil 50% saat pencarian wajah agar cepat (14 ms)
HOLD_FACE_SEC = 0.8          # Waktu toleransi mempertahankan kotak wajah jika deteksi miss sesaat
RESET_TIMEOUT_SEC = 2.0      # Waktu reset status jika wajah benar-benar meninggalkan kamera


# ==============================================================================
# 2. FUNGSI-FUNGSI PENDUKUNG (HELPER FUNCTIONS)
# ==============================================================================

def hitung_ear(eye_points):
    """
    Menghitung Eye Aspect Ratio (EAR) berdasarkan 6 titik koordinat kelopak mata.
    
    Formula Ilmiah (Soukupová & Čech, 2016):
                 ||p2 - p6|| + ||p3 - p5||
        EAR = -----------------------------
                     2.0 * ||p1 - p4||

    Penjelasan untuk Orang Awam:
    - p1 dan p4 adalah sudut kiri dan kanan mata (lebar mata).
    - p2, p6 dan p3, p5 adalah jarak vertikal kelopak atas ke kelopak bawah (tinggi mata).
    - Ketika mata terbuka lebar, tinggi mata besar -> nilai EAR tinggi (> 0.25).
    - Ketika mata berkedip/terpejam, tinggi mata mendekati 0 -> nilai EAR anjlok (< 0.22).
    """
    pts = np.array(eye_points, dtype=float)
    # Jarak vertikal 1 (kelopak atas ke kelopak bawah sisi dalam)
    vert_1 = np.linalg.norm(pts[1] - pts[5])
    # Jarak vertikal 2 (kelopak atas ke kelopak bawah sisi luar)
    vert_2 = np.linalg.norm(pts[2] - pts[4])
    # Jarak horizontal (sudut mata luar ke sudut mata dalam)
    horiz = np.linalg.norm(pts[0] - pts[3])

    if horiz == 0.0:
        return 0.0
    return (vert_1 + vert_2) / (2.0 * horiz)


def muat_database(file_path):
    """
    Memuat database embedding wajah yang tersimpan di disk.
    Toleran terhadap nama file lama (misal 'en   codings.pickle' yang ada spasi).
    """
    targets = [file_path, "en   codings.pickle", "encodings.pickle"]
    for path in targets:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, list):
                    print(f"[INFO] Berhasil memuat {len(data)} sampel wajah dari: {path}")
                    return data, path
                elif isinstance(data, dict) and "encodings" in data:
                    print(f"[INFO] Berhasil memuat {len(data['encodings'])} sampel dari dict: {path}")
                    return data["encodings"], path
            except Exception as e:
                print(f"[WARNING] Gagal membaca file {path}: {e}")
    print("[INFO] Database wajah belum ditemukan. Silakan daftarkan wajah dengan tombol 'e'.")
    return [], file_path


def buka_kamera(index_pilihan):
    """
    Membuka kamera dengan penanganan DirectShow di Windows.
    Otomatis fallback ke kamera index 0 jika index pilihan tidak bisa dibuka.
    """
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index_pilihan, backend)

    # Jika index pilihan gagal, coba index 0 (kamera bawaan laptop)
    if not cap.isOpened() and index_pilihan != 0:
        print(f"[WARNING] Kamera index {index_pilihan} gagal dibuka, mencoba index 0...")
        cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        raise SystemExit(
            "[ERROR] Kamera tidak dapat diakses! Pastikan webcam terpasang dan tidak "
            "sedang digunakan oleh aplikasi lain (seperti Zoom, browser, dsb)."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def gambar_bar_ear(frame, ear_val, threshold, x=15, y=410, w=180, h=18):
    """
    Menggambar progress bar visual nilai EAR di layar secara real-time.
    Sangat berguna untuk demonstrasi sidang skripsi agar penguji melihat
    grafik penurunan bukaan mata saat mata berkedip.
    """
    max_ear = 0.40
    clamped_ear = min(max(ear_val, 0.0), max_ear)
    fill_w = int((clamped_ear / max_ear) * w)

    # Warna bar: Hijau jika melek, Merah jika di bawah batas kedip
    bar_color = (0, 220, 0) if ear_val >= threshold else (0, 0, 255)

    # Background bar
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    # Fill bar
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), bar_color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)

    # Garis penanda batas threshold (kuning)
    thresh_x = x + int((threshold / max_ear) * w)
    cv2.line(frame, (thresh_x, y - 3), (thresh_x, y + h + 3), (0, 255, 255), 2)

    # Keterangan teks
    cv2.putText(frame, f"EAR: {ear_val:.2f} (Thresh: {threshold:.2f})",
                (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


# ==============================================================================
# 3. INISIALISASI VARIABEL & DATABASE
# ==============================================================================

# Buka koneksi video kamera
cap = buka_kamera(CAM_INDEX)

# Muat embedding wajah yang sudah terdaftar
known_encodings, active_db_path = muat_database(FILE_DATABASE)

# Variabel State Tracking
frame_count = 0
last_detected_locs = []     # Lokasi wajah pada frame terkini
tracked_loc = None          # Lokasi wajah yang dihaluskan (smoothed)
last_seen_time = 0.0        # Waktu terakhir wajah terlihat kamera

# Variabel Liveness (Blink Detection)
closed_counter = 0          # Menghitung berapa frame mata berturut-turut terpejam
blink_timestamps = []       # Catatan waktu terjadinya kedipan sah
current_ear = 0.0           # Nilai EAR rata-rata mata kiri & kanan saat ini

# Variabel Pengenalan Wajah
recognized_name = "Unknown" # Nama hasil pencocokan wajah
face_distance_val = 1.0     # Skor jarak kemiripan
last_recog_time = 0.0       # Timestamp terakhir kali kalkulasi encoding dijalankan

# Variabel Tampilan & Notifikasi
pesan_notifikasi = ""
pesan_until_time = 0.0
fps = 0.0
t_prev = time.time()
detection_time_ms = 0.0

print("\n" + "="*55)
print(" SISTEM PENGENALAN WAJAH & LIVENESS SIAP DIGUNAKAN")
print(" Tombol Kontrol:")
print("   [e] : Daftarkan / Tambah sampel wajah Anda")
print("   [r] : Reset status verifikasi")
print("   [q] : Keluar dari program")
print("="*55 + "\n")


# ==============================================================================
# 4. LOOP UTAMA PEMROSESAN VIDEO REAL-TIME
# ==============================================================================

while True:
    sukses, frame = cap.read()
    if not sukses:
        print("[WARNING] Frame kamera gagal dibaca.")
        time.sleep(0.05)
        continue

    frame_count += 1
    waktu_sekarang = time.time()

    # OpenCV membaca format BGR. dlib & face_recognition membutuhkan RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------------------------
    # TAHAP A: DETEKSI LOKASI WAJAH (FRAME SKIPPING UNTUK EFISIENSI CPU)
    # --------------------------------------------------------------------------
    jalankan_deteksi_baru = (frame_count % PROCESS_EVERY_N_FRAMES == 0) or (tracked_loc is None)

    if jalankan_deteksi_baru:
        t0 = time.time()
        # Kecilkan citra untuk mempercepat HOG Face Detector
        small_rgb = cv2.resize(rgb_frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
        
        # dlib HOG mencari kotak wajah pada gambar kecil
        locs_small = face_recognition.face_locations(small_rgb, number_of_times_to_upsample=0, model="hog")
        detection_time_ms = (time.time() - t0) * 1000.0

        # Kembalikan koordinat kotak wajah ke ukuran frame asli
        inv_scale = 1.0 / DETECT_SCALE
        last_detected_locs = [
            (int(top * inv_scale), int(right * inv_scale),
             int(bottom * inv_scale), int(left * inv_scale))
            for (top, right, bottom, left) in locs_small
        ]

    wajah_ada_sekarang = len(last_detected_locs) > 0

    if wajah_ada_sekarang:
        # Ambil wajah terbesar (terdekat ke kamera)
        wajah_terbesar = max(last_detected_locs, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
        
        # Bounding Box Smoothing (mencegah kotak bergetar/jitter)
        if tracked_loc is None:
            tracked_loc = wajah_terbesar
        else:
            alpha = 0.65
            tracked_loc = (
                int(alpha * wajah_terbesar[0] + (1 - alpha) * tracked_loc[0]),
                int(alpha * wajah_terbesar[1] + (1 - alpha) * tracked_loc[1]),
                int(alpha * wajah_terbesar[2] + (1 - alpha) * tracked_loc[2]),
                int(alpha * wajah_terbesar[3] + (1 - alpha) * tracked_loc[3]),
            )
        last_seen_time = waktu_sekarang

    elif tracked_loc is not None and (waktu_sekarang - last_seen_time <= HOLD_FACE_SEC):
        # Tahan posisi jika luput sekejap (toleransi)
        pass
    else:
        tracked_loc = None

    # Reset otomatis jika orang meninggalkan kamera
    if waktu_sekarang - last_seen_time > RESET_TIMEOUT_SEC:
        recognized_name = "Unknown"
        closed_counter = 0
        current_ear = 0.0
        blink_timestamps.clear()

    # --------------------------------------------------------------------------
    # TAHAP B: EKSTRAKSI LANDMARK MATA & PERHITUNGAN EAR (LIVENESS)
    # --------------------------------------------------------------------------
    landmarks_mata_kiri = None
    landmarks_mata_kanan = None

    if tracked_loc is not None:
        landmarks_list = face_recognition.face_landmarks(rgb_frame, [tracked_loc])
        
        if landmarks_list:
            landmarks = landmarks_list[0]
            landmarks_mata_kiri = landmarks.get("left_eye")
            landmarks_mata_kanan = landmarks.get("right_eye")

            if landmarks_mata_kiri and landmarks_mata_kanan:
                ear_kiri = hitung_ear(landmarks_mata_kiri)
                ear_kanan = hitung_ear(landmarks_mata_kanan)
                current_ear = (ear_kiri + ear_kanan) / 2.0

                # Logika Siklus Kedipan (Terbuka -> Tertutup -> Terbuka)
                if current_ear < EAR_THRESHOLD:
                    closed_counter += 1
                else:
                    if MIN_CLOSED_FRAMES <= closed_counter <= MAX_CLOSED_FRAMES:
                        blink_timestamps.append(waktu_sekarang)
                        pesan_notifikasi = "Kedipan Valid Terdeteksi!"
                        pesan_until_time = waktu_sekarang + 1.2
                    closed_counter = 0

        # ----------------------------------------------------------------------
        # TAHAP C: PENGENALAN WAJAH (FACE RECOGNITION MATCHING)
        # ----------------------------------------------------------------------
        if known_encodings and (waktu_sekarang - last_recog_time > RECOG_INTERVAL_SEC):
            encodings = face_recognition.face_encodings(rgb_frame, [tracked_loc])
            if encodings:
                distances = face_recognition.face_distance(known_encodings, encodings[0])
                face_distance_val = float(np.min(distances))

                if face_distance_val <= TOLERANCE:
                    recognized_name = NAMA_PENGGUNA
                else:
                    recognized_name = "Unknown"

            last_recog_time = waktu_sekarang

    # Hapus kedipan yang sudah kadaluarsa
    blink_timestamps = [t for t in blink_timestamps if (waktu_sekarang - t) <= BLINK_VALID_SEC]
    kedipan_valid_count = len(blink_timestamps)

    # --------------------------------------------------------------------------
    # TAHAP D: STATE MACHINE PENENTUAN STATUS AKHIR
    # --------------------------------------------------------------------------
    if tracked_loc is None:
        status_teks = "ARAHKAN WAJAH KE DEPAN KAMERA"
        warna_status = (160, 160, 160)
        status_detail = "Menunggu subjek wajah terdeteksi..."
    elif not known_encodings:
        status_teks = "DATABASE KOSONG - TEKAN 'e' UNTUK DAFTAR"
        warna_status = (0, 165, 255)
        status_detail = f"Tekan 'e' 3-5 kali untuk merekam sampel {NAMA_PENGGUNA}"
    elif recognized_name == "Unknown":
        status_teks = "AKSES DITOLAK: WAJAH TIDAK DIKENAL"
        warna_status = (0, 0, 255)
        status_detail = f"Wajah tidak terdaftar (Dist: {face_distance_val:.2f} > Tol: {TOLERANCE:.2f})"
    elif kedipan_valid_count >= REQUIRED_BLINKS:
        status_teks = f"AKSES DITERIMA: {recognized_name}"
        warna_status = (0, 220, 0)
        status_detail = f"Liveness Terverifikasi (Kedip: {kedipan_valid_count}x | Dist: {face_distance_val:.2f})"
    else:
        status_teks = f"WAJAH COCOK: {recognized_name}"
        warna_status = (0, 215, 255)
        status_detail = "SILAKAN BERKEDIP 1x UNTUK VERIFIKASI KEASLIAN (LIVENESS)"

    # --------------------------------------------------------------------------
    # TAHAP E: PENGGAMBARAN GRAFIS & ANTARMUKA PENGGUNA (UI / OSD)
    # --------------------------------------------------------------------------

    # 1. Kotak Wajah
    if tracked_loc is not None:
        top, right, bottom, left = tracked_loc
        cv2.rectangle(frame, (left, top), (right, bottom), warna_status, 2)
        label = recognized_name if recognized_name != "Unknown" else "Tidak Dikenal"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (left, bottom), (left + label_size[0] + 10, bottom + 25), warna_status, -1)
        cv2.putText(frame, label, (left + 5, bottom + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 2. Titik Landmark Mata (6 Titik Mata Kiri & Kanan)
    for eye in (landmarks_mata_kiri, landmarks_mata_kanan):
        if eye:
            for pt in eye:
                cv2.circle(frame, pt, 2, (255, 255, 0), -1)

    # 3. Bar Header Status di Bagian Atas
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 65), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (0, 62), (FRAME_WIDTH, 65), warna_status, -1)

    cv2.putText(frame, status_teks, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, warna_status, 2)
    cv2.putText(frame, status_detail, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

    # 4. Bar Indikator EAR di Bagian Bawah
    gambar_bar_ear(frame, current_ear, EAR_THRESHOLD, x=15, y=435, w=170, h=16)

    # 5. Informasi Metrik Sistem
    t_now = time.time()
    dt = t_now - t_prev
    t_prev = t_now
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

    cv2.putText(frame, f"FPS: {fps:.1f}", (FRAME_WIDTH - 110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"HOG: {detection_time_ms:.0f} ms", (FRAME_WIDTH - 110, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    info_bawah = f"Sampel DB: {len(known_encodings)} | Kedip: {kedipan_valid_count}"
    cv2.putText(frame, info_bawah, (FRAME_WIDTH - 240, 448),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    # Notifikasi Pop-up
    if waktu_sekarang < pesan_until_time:
        popup_w = 400
        popup_x = (FRAME_WIDTH - popup_w) // 2
        cv2.rectangle(frame, (popup_x, 80), (popup_x + popup_w, 115), (0, 140, 255), -1)
        cv2.putText(frame, pesan_notifikasi, (popup_x + 15, 103),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # --------------------------------------------------------------------------
    # TAHAP F: TAMPILKAN KE LAYAR & PENANGANAN TOMBOL KEYBOARD
    # --------------------------------------------------------------------------
    cv2.imshow("Autentikasi Wajah + Liveness Detection (Skripsi)", frame)
    key = cv2.waitKey(1) & 0xFF

    # [q] KELUAR
    if key == ord("q") or key == 27:
        print("[INFO] Pengguna menghentikan program.")
        break

    # [e] DAFTARKAN SAMPEL WAJAH BARU KE DATABASE
    elif key == ord("e"):
        if tracked_loc is not None:
            sampel_enc = face_recognition.face_encodings(rgb_frame, [tracked_loc])
            if sampel_enc:
                known_encodings.append(sampel_enc[0])
                with open(FILE_DATABASE, "wb") as f:
                    pickle.dump(known_encodings, f)
                pesan_notifikasi = f"Sampel {len(known_encodings)} tersimpan untuk {NAMA_PENGGUNA}!"
                pesan_until_time = waktu_sekarang + 2.5
                print(f"[SUKSES] Sampel wajah ke-{len(known_encodings)} berhasil disimpan.")
            else:
                pesan_notifikasi = "Gagal mengambil fitur wajah. Coba lagi!"
                pesan_until_time = waktu_sekarang + 2.0
        else:
            pesan_notifikasi = "Tidak ada wajah untuk didaftarkan!"
            pesan_until_time = waktu_sekarang + 2.0

    # [r] RESET VERIFIKASI
    elif key == ord("r"):
        blink_timestamps.clear()
        closed_counter = 0
        pesan_notifikasi = "Status verifikasi di-reset."
        pesan_until_time = waktu_sekarang + 1.5

# ==============================================================================
# 5. PEMBERSIHAN SUMBER DAYA (CLEANUP)
# ==============================================================================
cap.release()
cv2.destroyAllWindows()
print("[INFO] Program selesai dan kamera telah ditutup dengan aman.")