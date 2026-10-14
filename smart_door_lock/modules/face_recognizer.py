# ===============================================
# MODULE: FACE RECOGNIZER
# Fungsi: Face encoding dan recognition dari database
# Library: face_recognition
# ===============================================

import face_recognition
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
import cv2
import config


class FaceRecognizer:
    """
    Class untuk mengenali wajah dari database lokal.
    Menggunakan library face_recognition yang berbasis model dlib.
    """
    
    def __init__(self, database_path: str = config.FACE_DATABASE_PATH):
        """
        Inisialisasi Face Recognizer dan load semua encoding dari database.
        
        Args:
            database_path: Path ke folder berisi foto wajah yang sah
        """
        
        self.database_path = database_path
        self.known_face_encodings = []  # List berisi face encoding dari database
        self.known_face_names = []       # List berisi nama person dari database
        
        # Load encoding dari database
        self._load_face_database()
        
        if config.DEBUG_MODE:
            print(f"[FaceRecognizer] Loaded {len(self.known_face_names)} person(s) dari database")
    
    
    def _load_face_database(self):
        """
        Load semua gambar dari database folder dan encode wajahnya.
        
        Expected folder structure:
        face_database/
        ├── person_1.jpg
        ├── person_1_2.jpg
        ├── person_2.jpg
        └── ...
        
        Nama person di-extract dari nama file (sebelum angka/underscore)
        Contoh: "john_smith.jpg" -> person = "john_smith"
        """
        
        if not os.path.exists(self.database_path):
            print(f"[WARNING] Database path '{self.database_path}' tidak ditemukan!")
            print(f"[INFO] Silakan buat folder dan masukkan foto wajah")
            return
        
        # Supported extensions
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Load semua gambar dari folder
        for filename in os.listdir(self.database_path):
            file_path = os.path.join(self.database_path, filename)
            
            # Skip jika bukan file gambar
            if not os.path.isfile(file_path):
                continue
            
            if not any(filename.lower().endswith(ext) for ext in supported_formats):
                continue
            
            try:
                # Load gambar
                image = face_recognition.load_image_file(file_path)
                
                # Detect wajah di dalam gambar
                face_locations = face_recognition.face_locations(image)
                
                if len(face_locations) == 0:
                    print(f"[WARNING] Tidak ada wajah terdeteksi di {filename}")
                    continue
                
                # Jika ada multiple faces, gunakan yang terbesar (face pertama)
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                
                # Extract nama person dari filename
                # Format: "john_smith.jpg" atau "john_smith_1.jpg" -> "john_smith"
                name_without_ext = os.path.splitext(filename)[0]
                person_name = self._extract_person_name(name_without_ext)
                
                # Simpan encoding dan nama
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(person_name)
                
                if config.DEBUG_MODE:
                    print(f"[INFO] Loaded: {filename} -> {person_name}")
                    
            except Exception as e:
                print(f"[ERROR] Gagal memproses {filename}: {str(e)}")
    
    
    def _extract_person_name(self, filename_without_ext: str) -> str:
        """
        Extract nama person dari filename.
        
        Contoh:
        - "john_smith" -> "john_smith"
        - "john_smith_1" -> "john_smith"
        - "john_smith_photo" -> "john_smith_photo"
        
        Args:
            filename_without_ext: Nama file tanpa extension
            
        Returns:
            Nama person
        """
        
        # Coba remove trailing numbers yang dipisahkan underscore atau minus
        parts = filename_without_ext.rsplit('_', 1)
        
        if len(parts) == 2:
            # Cek apakah bagian belakang adalah number
            if parts[1].isdigit():
                return parts[0]
        
        return filename_without_ext
    
    
    def encode_face(self, frame: np.ndarray, face_location: Tuple) -> Optional[np.ndarray]:
        """
        Generate face encoding dari face region dalam frame.
        
        Args:
            frame: Input frame (RGB format)
            face_location: Tuple (top, right, bottom, left) dari bounding box
            
        Returns:
            Face encoding (128-d vector) atau None jika gagal
        """
        
        try:
            face_encodings = face_recognition.face_encodings(frame, [face_location])
            
            if len(face_encodings) > 0:
                return face_encodings[0]
            else:
                return None
                
        except Exception as e:
            print(f"[ERROR] Gagal encode face: {str(e)}")
            return None
    
    
    def recognize_face(self, face_encoding: np.ndarray) -> Tuple[str, float]:
        """
        Recognize wajah dengan membandingkan terhadap database.
        
        Args:
            face_encoding: Face encoding dari wajah yang akan dikenali
            
        Returns:
            Tuple (person_name, confidence)
            - person_name: Nama orang yang paling mirip (atau "Unknown" jika tidak ada)
            - confidence: Nilai confidence (0-1, semakin tinggi semakin mirip)
        """
        
        # Jika database kosong
        if len(self.known_face_encodings) == 0:
            return "Unknown", 0.0
        
        # Compare dengan semua encoding di database
        face_distances = face_recognition.face_distance(
            self.known_face_encodings, 
            face_encoding
        )
        
        # Cari match terdekat
        best_match_index = np.argmin(face_distances)
        best_match_distance = face_distances[best_match_index]
        
        # Confidence = 1 - distance (semakin kecil distance, semakin tinggi confidence)
        confidence = 1.0 - best_match_distance
        
        # Cek apakah match memenuhi tolerance threshold
        if best_match_distance < config.FACE_RECOGNITION_TOLERANCE:
            person_name = self.known_face_names[best_match_index]
            return person_name, confidence
        else:
            return "Unknown", confidence
    
    
    def get_access_result(self, person_name: str) -> Tuple[str, Tuple[int, int, int]]:
        """
        Dapatkan hasil access berdasarkan person name yang dikenali.
        
        Args:
            person_name: Nama person dari recognition
            
        Returns:
            Tuple (message, color) dalam format BGR
        """
        
        if person_name == "Unknown":
            return ("AKSES DITOLAK - Wajah Tidak Dikenali", config.COLOR_RED)
        else:
            return (f"PINTU TERBUKA - Selamat Datang {person_name}!", config.COLOR_GREEN)
    
    
    def get_database_info(self) -> Dict:
        """
        Dapatkan informasi tentang database yang ter-load.
        
        Returns:
            Dictionary dengan info database
        """
        
        # Count unique persons
        unique_persons = len(set(self.known_face_names))
        
        # Count total photos
        total_photos = len(self.known_face_encodings)
        
        # Grouping by person
        person_counts = {}
        for name in self.known_face_names:
            person_counts[name] = person_counts.get(name, 0) + 1
        
        return {
            'database_path': self.database_path,
            'total_persons': unique_persons,
            'total_photos': total_photos,
            'person_counts': person_counts
        }
    
    
    def is_database_empty(self) -> bool:
        """Cek apakah database kosong."""
        return len(self.known_face_encodings) == 0
