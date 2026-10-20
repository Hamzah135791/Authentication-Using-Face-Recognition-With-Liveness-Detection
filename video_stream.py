"""
================================================================================
MODUL 1: MULTITHREADING CAMERA STREAM (RASPBERRY PI OPTIMIZED)
================================================================================
Kelas VideoStreamThread memisahkan proses pembacaan frame kamera I/O ke
thread latar belakang (daemon thread). Hal ini mencegah operasi I/O kamera
yang lambat dari memblokir (blocking) loop kalkulasi visi komputer utama.

Fitur Optimasi:
1. Non-blocking read: Loop utama selalu mendapatkan frame termutakhir (0 ms wait).
2. Buffer size = 1: Mencegah penumpukan frame usang di buffer kernel.
3. Otomatis Downscaling ke 320x240: Memangkas 75% beban piksel CPU Raspberry Pi 3.
"""

import os
import time
import threading
import cv2


class VideoStreamThread:
    """
    Kelas penangkap stream kamera multi-threading.
    """
    def __init__(self, src=0, width=320, height=240, fps=30):
        self.src = src
        self.target_width = width
        self.target_height = height
        self.target_fps = fps

        # Pemilihan backend kamera berdasarkan sistem operasi
        # Di Linux / Raspberry Pi: V4L2 (Video for Linux 2) sangat efisien
        # Di Windows: DirectShow (CAP_DSHOW)
        if os.name == "nt":
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_V4L2

        # Inisialisasi koneksi kamera
        print(f"[INFO] Membuka kamera (Index: {self.src})...")
        self.cap = cv2.VideoCapture(self.src, backend)

        # Fallback jika backend tertentu gagal diinisialisasi
        if not self.cap.isOpened():
            print(f"[WARNING] Gagal membuka dengan backend spesifik, mencoba CAP_ANY...")
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened() and self.src != 0:
            print(f"[WARNING] Kamera index {self.src} tidak ditemukan. Mencoba index 0...")
            self.src = 0
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[ERROR] Kamera (index {self.src}) gagal dibuka! "
                "Pastikan kamera terhubung dan modul kamera Raspberry Pi (raspi-config) aktif."
            )

        # Konfigurasi properti hardware kamera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Mengurangi ukuran buffer OpenCV menjadi 1 frame untuk mencegah frame lag
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Baca frame awal pertama kali
        self.grabbed, raw_frame = self.cap.read()
        if not self.grabbed or raw_frame is None:
            raise RuntimeError("[ERROR] Tidak dapat membaca frame pertama dari kamera.")

        # Pastikan ukuran frame awal sesuai target (320x240)
        self.frame = self._preprocess_frame(raw_frame)

        # Status kontrol thread
        self.stopped = False
        self.lock = threading.Lock()

        # Thread pekerja latar belakang (daemon agar otomatis tertutup jika program mati)
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        """Memulai eksekusi background thread."""
        self.thread.start()
        print(f"[INFO] Thread kamera aktif. Resolusi capture: {self.target_width}x{self.target_height}")
        return self

    def _preprocess_frame(self, frame):
        """
        Memastikan ukuran frame selalu 320x240 untuk performa optimal Raspberry Pi.
        """
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            # Gunakan INTER_LINEAR untuk resize cepat dan efisien di CPU ARM
            return cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        return frame

    def _worker(self):
        """
        Fungsi loop pekerja di background thread yang terus membaca frame terbaru.
        """
        while not self.stopped:
            grabbed, raw_frame = self.cap.read()
            if not grabbed:
                # Beri jeda sangat kecil agar thread tidak membebani 1 core penuh saat kamera idle
                time.sleep(0.005)
                continue

            processed_frame = self._preprocess_frame(raw_frame)

            # Update frame secara thread-safe menggunakan lock
            with self.lock:
                self.grabbed = grabbed
                self.frame = processed_frame

    def read(self):
        """
        Mengambil salinan frame terbaru tanpa memblokir loop pemrosesan utama.
        Mengembalikan tuple: (grabbed: bool, frame: np.ndarray)
        """
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            # Return copy agar thread-safe dan tidak ada konflik saat frame diproses
            return True, self.frame.copy()

    def stop(self):
        """Menghentikan thread dan melepaskan sumber daya perangkat kamera."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        print("[INFO] Thread kamera berhasil dihentikan.")
