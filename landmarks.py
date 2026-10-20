"""
================================================================================
MODUL 2: LANDMARK DETECTION & EYE ASPECT RATIO (EAR) - LIVENESS
================================================================================
Modul ini mengimplementasikan arsitektur modular untuk deteksi landmark wajah
dan kalkulasi Eye Aspect Ratio (EAR) sebagai penentu keaslian (liveness).

Arsitektur:
1. BaseLandmarkDetector (Abstract Base Class / Interface)
2. MediaPipeLandmarkDetector (Default - Model FaceMesh TFLite sangat ringan di CPU)
3. DlibLandmarkDetector (Alternatif pembanding skripsi - Dlib 68 Landmarks)
4. hitung_ear() (Kalkulasi ilmiah EAR Soukupová & Čech, 2016)
"""

import os
import time
import abc
import numpy as np
import cv2


def hitung_ear(eye_points):
    """
    Menghitung Eye Aspect Ratio (EAR) berdasarkan 6 titik koordinat mata.
    Formula Ilmiah (Soukupová & Čech, 2016):
                 ||p2 - p6|| + ||p3 - p5||
        EAR = -----------------------------
                     2.0 * ||p1 - p4||

    Parameter:
        eye_points: List atau np.ndarray berisi 6 titik koordinat (x, y)
                    dengan urutan [p1, p2, p3, p4, p5, p6].
    Return:
        ear: float nilai rasio bukaan mata (biasanya 0.25 - 0.35 melek, < 0.22 terpejam)
    """
    pts = np.array(eye_points, dtype=np.float32)
    # Jarak vertikal 1 (kelopak atas - kelopak bawah bagian tepi)
    vert_1 = np.linalg.norm(pts[1] - pts[5])
    # Jarak vertikal 2 (kelopak atas - kelopak bawah bagian tengah/dalam)
    vert_2 = np.linalg.norm(pts[2] - pts[4])
    # Jarak horizontal (sudut mata luar ke sudut mata dalam)
    horiz = np.linalg.norm(pts[0] - pts[3])

    if horiz == 0.0:
        return 0.0
    return float((vert_1 + vert_2) / (2.0 * horiz))


class LandmarkResult:
    """Struktur data hasil deteksi landmark wajah."""
    def __init__(self):
        self.face_detected = False
        self.bbox = None            # (x1, y1, x2, y2) koordinat kotak wajah
        self.face_crop = None        # Crop area wajah (BGR)
        self.left_eye = []           # 6 titik mata kiri [(x, y), ...]
        self.right_eye = []          # 6 titik mata kanan [(x, y), ...]
        self.all_landmarks = []      # Semua titik landmark (opsional)
        self.ear_left = 0.0
        self.ear_right = 0.0
        self.ear_avg = 0.0
        self.inference_time_ms = 0.0 # Waktu pemrosesan model (milidetik)


class BaseLandmarkDetector(abc.ABC):
    """
    Kelas abstrak dasar (Interface) untuk detektor landmark wajah.
    Memungkinkan penggantian model landmark secara 'plug-and-play'.
    """
    @abc.abstractmethod
    def detect(self, frame_bgr) -> LandmarkResult:
        """Mendeteksi wajah dan landmark dari frame gambar BGR."""
        pass

    @abc.abstractmethod
    def close(self):
        """Membersihkan sumber daya model."""
        pass


class MediaPipeLandmarkDetector(BaseLandmarkDetector):
    """
    Implementasi detektor landmark menggunakan MediaPipe Face Mesh.
    Mendukung MediaPipe legacy (<0.10.15) via `solutions.face_mesh`
    dan MediaPipe modern (>=0.10.20 / 1.x) via Tasks API `FaceLandmarker`.
    """
    # Indeks titik landmark mata standar MediaPipe 468 titik untuk 6-titik EAR:
    # Mata Kanan (perspektif kamera / sisi kiri penonton)
    RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    # Mata Kiri (perspektif kamera / sisi kanan penonton)
    LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    def __init__(self, max_num_faces=1, refine_landmarks=False, task_model_path=None):
        try:
            import mediapipe as mp
            self.mp = mp
        except ImportError:
            raise ImportError(
                "[ERROR] Library 'mediapipe' belum terinstal! "
                "Silakan jalankan: pip install mediapipe"
            )

        self.max_num_faces = max_num_faces
        self.face_mesh = None
        self.landmarker = None

        # Deteksi versi API MediaPipe: legacy solutions vs modern Tasks API
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.mode = "solutions"
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[INFO] MediaPipe Face Mesh (Solutions API) berhasil diinisialisasi.")
        else:
            self.mode = "tasks"
            if task_model_path is None:
                task_model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "models",
                    "face_landmarker.task"
                )
            self._ensure_task_model(task_model_path)

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=task_model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=max_num_faces,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            print("[INFO] MediaPipe Face Landmarker (Tasks API) berhasil diinisialisasi.")

    def _ensure_task_model(self, path):
        """Memastikan model .task MediaPipe tersedia di lokal, jika belum ada otomatis unduh."""
        if not os.path.exists(path):
            import urllib.request
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            print(f"[INFO] Mengunduh model MediaPipe Tasks ({os.path.basename(path)})...")
            urllib.request.urlretrieve(url, path)
            print("[INFO] Model MediaPipe Tasks berhasil diunduh.")

    def detect(self, frame_bgr) -> LandmarkResult:
        result = LandmarkResult()
        if frame_bgr is None:
            return result

        h, w = frame_bgr.shape[:2]
        # MediaPipe membutuhkan citra berformat RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Catat waktu awal inferensi
        t_start = time.perf_counter()

        if self.mode == "solutions":
            mesh_results = self.face_mesh.process(frame_rgb)
            t_end = time.perf_counter()
            result.inference_time_ms = (t_end - t_start) * 1000.0

            if not mesh_results.multi_face_landmarks:
                return result
            face_lms = mesh_results.multi_face_landmarks[0].landmark
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
            task_results = self.landmarker.detect(mp_image)
            t_end = time.perf_counter()
            result.inference_time_ms = (t_end - t_start) * 1000.0

            if not task_results.face_landmarks:
                return result
            face_lms = task_results.face_landmarks[0]

        result.face_detected = True

        # Ekstraksi koordinat semua titik dalam satuan piksel
        coords = []
        for lm in face_lms:
            px = int(lm.x * w)
            py = int(lm.y * h)
            coords.append((px, py))
        result.all_landmarks = coords

        # Hitung bounding box wajah dengan sedikit padding
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
        y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))

        # Tambahkan margin padding 10% agar area dahi dan dagu tercakup
        pad_x = int((x_max - x_min) * 0.10)
        pad_y = int((y_max - y_min) * 0.15)
        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        result.bbox = (x1, y1, x2, y2)
        if x2 > x1 and y2 > y1:
            result.face_crop = frame_bgr[y1:y2, x1:x2].copy()

        # Ekstraksi 6 titik mata kanan & kiri
        result.right_eye = [coords[idx] for idx in self.RIGHT_EYE_IDX]
        result.left_eye = [coords[idx] for idx in self.LEFT_EYE_IDX]

        # Hitung rasio EAR untuk masing-masing mata
        result.ear_right = hitung_ear(result.right_eye)
        result.ear_left = hitung_ear(result.left_eye)
        result.ear_avg = (result.ear_right + result.ear_left) / 2.0

        return result

    def close(self):
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
        if hasattr(self, 'landmarker') and self.landmarker is not None:
            self.landmarker.close()


class DlibLandmarkDetector(BaseLandmarkDetector):
    """
    Implementasi detektor landmark menggunakan Dlib (HOG Face Detector + 68 Landmarks).
    Disediakan sebagai modul pembanding dalam pengujian performa Skripsi.
    """
    def __init__(self, predictor_path=None):
        try:
            import dlib
        except ImportError:
            raise ImportError(
                "[ERROR] Library 'dlib' belum terinstal! "
                "Silakan instal dlib atau gunakan MediaPipe."
            )

        if predictor_path is None or not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"[ERROR] File landmark predictor '{predictor_path}' tidak ditemukan! "
                "Unduh shape_predictor_68_face_landmarks.dat ke folder models/."
            )

        self.dlib = dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        print(f"[INFO] Dlib 68-Landmark Detector berhasil diinisialisasi.")

    def detect(self, frame_bgr) -> LandmarkResult:
        result = LandmarkResult()
        if frame_bgr is None:
            return result

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Catat waktu inferensi Dlib
        t_start = time.perf_counter()
        rects = self.detector(gray, 0) # 0 = tidak ada upsample (demi kecepatan)

        if len(rects) == 0:
            result.inference_time_ms = (time.perf_counter() - t_start) * 1000.0
            return result

        # Proses wajah pertama yang terdeteksi
        rect = rects[0]
        shape = self.predictor(gray, rect)
        t_end = time.perf_counter()
        result.inference_time_ms = (t_end - t_start) * 1000.0

        result.face_detected = True

        # Koordinat 68 titik
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        result.all_landmarks = coords

        x1 = max(0, rect.left())
        y1 = max(0, rect.top())
        x2 = min(w, rect.right())
        y2 = min(h, rect.bottom())
        result.bbox = (x1, y1, x2, y2)

        if x2 > x1 and y2 > y1:
            result.face_crop = frame_bgr[y1:y2, x1:x2].copy()

        # Dlib: Mata Kanan (indeks 36-41), Mata Kiri (indeks 42-47)
        result.right_eye = coords[36:42]
        result.left_eye = coords[42:48]

        result.ear_right = hitung_ear(result.right_eye)
        result.ear_left = hitung_ear(result.left_eye)
        result.ear_avg = (result.ear_right + result.ear_left) / 2.0

        return result

    def close(self):
        pass


def get_landmark_detector(backend="mediapipe", predictor_path=None, task_model_path=None) -> BaseLandmarkDetector:
    """
    Factory function untuk menginisialisasi detektor landmark yang dipilih.
    """
    backend = backend.lower()
    if backend == "mediapipe":
        return MediaPipeLandmarkDetector(task_model_path=task_model_path)
    elif backend == "dlib":
        return DlibLandmarkDetector(predictor_path=predictor_path)
    else:
        raise ValueError(f"[ERROR] Detektor landmark '{backend}' tidak dikenali. Pilih: 'mediapipe' atau 'dlib'.")
