"""
================================================================================
MODUL 3: MODULAR FACE RECOGNITION (LBPH, MOBILEFACENET TFLITE, & DLIB)
================================================================================
Modul ini mengimplementasikan arsitektur pengenalan wajah modular untuk
memungkinkan komparasi metode pada penelitian/skripsi:

1. LBPHFaceRecognizer:
   - Menggunakan Local Binary Patterns Histograms (OpenCV).
   - Waktu inferensi ultra-cepat (< 5 ms di Raspberry Pi 3 CPU).
   - Sangat hemat memori RAM (< 10 MB).
2. MobileFaceNetRecognizer:
   - Deep CNN teroptimasi untuk perangkat bergerak/edge computing via TFLite.
   - Fitur 128-D/512-D embedding dengan tingkat akurasi tinggi (> 99%).
3. DlibFaceRecognizer:
   - Pembanding klasik deep metric learning 128-D menggunakan dlib/face_recognition.
   - Kompatibel dengan database lama (encodings.pickle).
"""

import os
import json
import pickle
import time
import abc
import numpy as np
import cv2


class RecognitionResult:
    """Struktur data hasil pengenalan wajah."""
    def __init__(self, name="Unknown", distance=1.0, is_match=False, inference_time_ms=0.0):
        self.name = name
        self.distance = distance
        self.is_match = is_match
        self.inference_time_ms = inference_time_ms


class BaseFaceRecognizer(abc.ABC):
    """Kelas abstrak dasar (Interface) untuk model Face Recognition."""
    @abc.abstractmethod
    def enroll(self, face_bgr, name: str) -> bool:
        """Mendaftarkan sampel wajah baru ke dalam database."""
        pass

    @abc.abstractmethod
    def recognize(self, face_bgr) -> RecognitionResult:
        """Mengenali identitas dari potongan citra wajah (face crop)."""
        pass

    @abc.abstractmethod
    def save_database(self):
        """Menyimpan database model ke penyimpanan persisten."""
        pass

    @abc.abstractmethod
    def load_database(self) -> bool:
        """Memuat database model dari disk."""
        pass

    @abc.abstractmethod
    def get_sample_count(self) -> int:
        """Mengembalikan jumlah sampel wajah terdaftar."""
        pass


class LBPHFaceRecognizer(BaseFaceRecognizer):
    """
    Implementasi Face Recognition berbasis Local Binary Patterns Histograms (LBPH).
    Sangat direkomendasikan sebagai pilihan default untuk Raspberry Pi 3 karena
    komputasinya berbasis tekstur lokal yang sangat ringan tanpa overhead Deep Learning.
    """
    def __init__(self, model_path=None, labels_path=None, distance_threshold=65.0):
        self.model_path = model_path
        self.labels_path = labels_path
        self.distance_threshold = distance_threshold

        # Validasi ketersediaan modul opencv-contrib
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            raise ImportError(
                "[ERROR] OpenCV LBPH tidak tersedia! Pastikan Anda menginstal package "
                "'opencv-contrib-python' (bukan opencv-python standar). "
                "Jalankan: pip install opencv-contrib-python"
            )

        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        self.labels_map = {}  # {label_id: name}
        self.face_samples = [] # [np.ndarray (100x100 gray)]
        self.face_labels = []  # [label_id]
        self.is_trained = False

        self.load_database()

    def _preprocess_face(self, face_bgr):
        """Pre-processing citra wajah: grayscale, resize 100x100, dan equalizeHist."""
        if face_bgr is None or face_bgr.size == 0:
            return None
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_LINEAR)
        # Normalisasi pencahayaan (Histogram Equalization)
        equalized = cv2.equalizeHist(resized)
        return equalized

    def enroll(self, face_bgr, name: str) -> bool:
        processed = self._preprocess_face(face_bgr)
        if processed is None:
            return False

        # Tentukan label ID numerik untuk nama pengguna
        rev_map = {v: k for k, v in self.labels_map.items()}
        if name in rev_map:
            label_id = rev_map[name]
        else:
            label_id = len(self.labels_map) + 1
            self.labels_map[label_id] = name

        self.face_samples.append(processed)
        self.face_labels.append(label_id)

        # Latih ulang recognizer dengan semua sampel yang ada
        self.recognizer.train(self.face_samples, np.array(self.face_labels, dtype=np.int32))
        self.is_trained = True
        self.save_database()
        print(f"[INFO] LBPH: Berhasil mendaftarkan sampel wajah ke-{len(self.face_samples)} untuk '{name}' (ID: {label_id}).")
        return True

    def recognize(self, face_bgr) -> RecognitionResult:
        result = RecognitionResult()
        if not self.is_trained or len(self.face_samples) == 0:
            return result

        processed = self._preprocess_face(face_bgr)
        if processed is None:
            return result

        t_start = time.perf_counter()
        label_id, distance = self.recognizer.predict(processed)
        t_end = time.perf_counter()

        result.inference_time_ms = (t_end - t_start) * 1000.0
        result.distance = float(distance)

        # Jarak LBPH: Semakin kecil semakin mirip. Jika di bawah threshold, cocok.
        if distance <= self.distance_threshold and label_id in self.labels_map:
            result.name = self.labels_map[label_id]
            result.is_match = True
        else:
            result.name = "Unknown"
            result.is_match = False

        return result

    def save_database(self):
        try:
            if self.model_path:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.recognizer.save(self.model_path)
            if self.labels_path:
                os.makedirs(os.path.dirname(self.labels_path), exist_ok=True)
                with open(self.labels_path, "w") as f:
                    # Serialisasi {str(id): name}
                    json.dump({str(k): v for k, v in self.labels_map.items()}, f, indent=2)
            # Simpan juga cache sampel gambar untuk re-training
            samples_cache_path = self.model_path + ".samples.pkl"
            with open(samples_cache_path, "wb") as f:
                pickle.dump({"samples": self.face_samples, "labels": self.face_labels}, f)
        except Exception as e:
            print(f"[WARNING] Gagal menyimpan model LBPH: {e}")

    def load_database(self) -> bool:
        if self.model_path and os.path.exists(self.model_path) and \
           self.labels_path and os.path.exists(self.labels_path):
            try:
                self.recognizer.read(self.model_path)
                with open(self.labels_path, "r") as f:
                    raw_map = json.load(f)
                    self.labels_map = {int(k): v for k, v in raw_map.items()}

                samples_cache_path = self.model_path + ".samples.pkl"
                if os.path.exists(samples_cache_path):
                    with open(samples_cache_path, "rb") as f:
                        data = pickle.load(f)
                        self.face_samples = data.get("samples", [])
                        self.face_labels = data.get("labels", [])

                self.is_trained = True
                print(f"[INFO] Berhasil memuat model LBPH ({len(self.face_samples)} sampel, {len(self.labels_map)} identitas).")
                return True
            except Exception as e:
                print(f"[WARNING] Gagal membaca model LBPH: {e}")
        return False

    def get_sample_count(self) -> int:
        return len(self.face_samples)


class MobileFaceNetRecognizer(BaseFaceRecognizer):
    """
    Implementasi Face Recognition menggunakan MobileFaceNet TFLite.
    Model CNN ini dirancang khusus untuk embedded edge hardware seperti Raspberry Pi.
    Menghasilkan 128-D atau 512-D L2-normalized embedding vector.
    """
    def __init__(self, model_path=None, db_path=None, threshold=0.40):
        self.model_path = model_path
        self.db_path = db_path
        self.threshold = threshold
        self.database = {} # {name: [embedding_vector, ...]}

        # Dukungan multi-runtime: Prioritaskan tflite_runtime (resmi untuk Raspberry Pi)
        self.interpreter = None
        self._init_interpreter()
        self.load_database()

    def _init_interpreter(self):
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"[INFO] File model MobileFaceNet '{self.model_path}' belum ditemukan.")
            print("[INFO] Silakan letakkan file 'mobilefacenet.tflite' di direktori models/.")
            return

        try:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite import Interpreter

            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("[INFO] MobileFaceNet TFLite Interpreter berhasil dimuat.")
        except Exception as e:
            print(f"[WARNING] Gagal menginisialisasi TFLite Interpreter: {e}")

    def _extract_embedding(self, face_bgr):
        """Ekstraksi embedding 128D/512D dari citra wajah menggunakan MobileFaceNet."""
        if self.interpreter is None or face_bgr is None or face_bgr.size == 0:
            return None

        # Input MobileFaceNet standar: 112x112 RGB
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)

        # Normalisasi citra ke range [-1.0, 1.0]
        input_data = (resized.astype(np.float32) - 127.5) / 128.0
        input_data = np.expand_dims(input_data, axis=0)

        # Inferensi TFLite
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # L2-normalization agar panjang vektor = 1.0
        norm = np.linalg.norm(raw_output)
        if norm == 0.0:
            return raw_output
        return raw_output / norm

    def enroll(self, face_bgr, name: str) -> bool:
        if self.interpreter is None:
            print("[ERROR] MobileFaceNet model belum dimuat, tidak dapat mendaftarkan wajah.")
            return False

        embedding = self._extract_embedding(face_bgr)
        if embedding is None:
            return False

        if name not in self.database:
            self.database[name] = []
        self.database[name].append(embedding)

        self.save_database()
        total_samples = sum(len(v) for v in self.database.values())
        print(f"[INFO] MobileFaceNet: Sampel wajah ke-{total_samples} tersimpan untuk '{name}'.")
        return True

    def recognize(self, face_bgr) -> RecognitionResult:
        result = RecognitionResult()
        if self.interpreter is None or len(self.database) == 0:
            return result

        t_start = time.perf_counter()
        probe_emb = self._extract_embedding(face_bgr)
        t_end = time.perf_counter()

        result.inference_time_ms = (t_end - t_start) * 1000.0
        if probe_emb is None:
            return result

        best_name = "Unknown"
        best_dist = 1.0 # Cosine distance range [0.0 (identik) s/d 2.0 (berlawanan)]

        for name, emb_list in self.database.items():
            for stored_emb in emb_list:
                # Cosine distance = 1 - dot_product(v1, v2)
                cosine_dist = 1.0 - float(np.dot(probe_emb, stored_emb))
                if cosine_dist < best_dist:
                    best_dist = cosine_dist
                    best_name = name

        result.distance = best_dist
        if best_dist <= self.threshold:
            result.name = best_name
            result.is_match = True
        else:
            result.name = "Unknown"
            result.is_match = False

        return result

    def save_database(self):
        if self.db_path:
            try:
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                with open(self.db_path, "wb") as f:
                    pickle.dump(self.database, f)
            except Exception as e:
                print(f"[WARNING] Gagal menyimpan database MobileFaceNet: {e}")

    def load_database(self) -> bool:
        if self.db_path and os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    self.database = pickle.load(f)
                total = sum(len(v) for v in self.database.values())
                print(f"[INFO] MobileFaceNet DB berhasil dimuat ({total} sampel, {len(self.database)} identitas).")
                return True
            except Exception as e:
                print(f"[WARNING] Gagal membaca MobileFaceNet DB: {e}")
        return False

    def get_sample_count(self) -> int:
        return sum(len(v) for v in self.database.values())


class DlibFaceRecognizer(BaseFaceRecognizer):
    """
    Implementasi Face Recognition menggunakan library face_recognition / dlib (128-D ResNet).
    Menjaga kompatibilitas dengan file 'encodings.pickle' yang telah dibuat sebelumnya.
    """
    def __init__(self, encodings_path=None, tolerance=0.50):
        self.encodings_path = encodings_path
        self.tolerance = tolerance
        self.known_encodings = []
        self.known_names = []

        try:
            import face_recognition
            self.fr = face_recognition
        except ImportError:
            self.fr = None
            print("[WARNING] Library 'face_recognition' tidak tersedia.")

        self.load_database()

    def enroll(self, face_bgr, name: str) -> bool:
        if self.fr is None or face_bgr is None or face_bgr.size == 0:
            return False

        h, w = face_bgr.shape[:2]
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        # Seluruh area face_crop dianggap sebagai kotak wajah
        loc = [(0, w, h, 0)] # (top, right, bottom, left)

        t_start = time.perf_counter()
        encs = self.fr.face_encodings(rgb, loc)
        if not encs:
            return False

        self.known_encodings.append(encs[0])
        self.known_names.append(name)
        self.save_database()
        print(f"[INFO] Dlib: Berhasil menambahkan sampel ke-{len(self.known_encodings)} untuk '{name}'.")
        return True

    def recognize(self, face_bgr) -> RecognitionResult:
        result = RecognitionResult()
        if self.fr is None or len(self.known_encodings) == 0 or face_bgr is None:
            return result

        h, w = face_bgr.shape[:2]
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        loc = [(0, w, h, 0)]

        t_start = time.perf_counter()
        encs = self.fr.face_encodings(rgb, loc)
        t_end = time.perf_counter()

        result.inference_time_ms = (t_end - t_start) * 1000.0
        if not encs:
            return result

        distances = self.fr.face_distance(self.known_encodings, encs[0])
        min_idx = int(np.argmin(distances))
        best_dist = float(distances[min_idx])

        result.distance = best_dist
        if best_dist <= self.tolerance:
            result.name = self.known_names[min_idx]
            result.is_match = True
        else:
            result.name = "Unknown"
            result.is_match = False

        return result

    def save_database(self):
        if self.encodings_path:
            try:
                with open(self.encodings_path, "wb") as f:
                    pickle.dump({"encodings": self.known_encodings, "names": self.known_names}, f)
            except Exception as e:
                print(f"[WARNING] Gagal menyimpan encodings dlib: {e}")

    def load_database(self) -> bool:
        if not self.encodings_path or not os.path.exists(self.encodings_path):
            return False
        try:
            with open(self.encodings_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                self.known_encodings = data
                self.known_names = ["Hamzah"] * len(data)
                print(f"[INFO] Memuat {len(self.known_encodings)} sampel dlib dari list format.")
                return True
            elif isinstance(data, dict) and "encodings" in data:
                self.known_encodings = data["encodings"]
                self.known_names = data.get("names", ["Hamzah"] * len(self.known_encodings))
                print(f"[INFO] Memuat {len(self.known_encodings)} sampel dlib dari dict format.")
                return True
        except Exception as e:
            print(f"[WARNING] Gagal membaca {self.encodings_path}: {e}")
        return False

    def get_sample_count(self) -> int:
        return len(self.known_encodings)


def get_face_recognizer(backend="lbph", **kwargs) -> BaseFaceRecognizer:
    """Factory function untuk inisialisasi modul Face Recognizer."""
    backend = backend.lower()
    if backend == "lbph":
        return LBPHFaceRecognizer(**kwargs)
    elif backend in ("mobilefacenet", "tflite"):
        return MobileFaceNetRecognizer(**kwargs)
    elif backend in ("dlib", "face_recognition"):
        return DlibFaceRecognizer(**kwargs)
    else:
        raise ValueError(
            f"[ERROR] Recognizer '{backend}' tidak didukung! "
            "Pilih: 'lbph', 'mobilefacenet', atau 'dlib'."
        )
