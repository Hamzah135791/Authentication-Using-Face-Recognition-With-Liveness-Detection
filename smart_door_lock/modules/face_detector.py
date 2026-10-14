# ===============================================
# MODULE: FACE DETECTOR
# Fungsi: Mendeteksi wajah dan mengekstrak facial landmarks
# Library: MediaPipe Face Mesh
# ===============================================

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional
import config


class FaceDetector:
    """
    Class untuk mendeteksi wajah dan facial landmarks menggunakan MediaPipe.
    MediaPipe lebih efisien untuk real-time processing, terutama di Raspberry Pi.
    """
    
    def __init__(self):
        """Inisialisasi MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=config.MEDIAPIPE_STATIC_IMAGE_MODE,
            max_num_faces=config.MEDIAPIPE_MAX_NUM_FACES,
            min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 255, 0)
        )
        
        if config.DEBUG_MODE:
            print("[FaceDetector] Inisialisasi MediaPipe Face Mesh berhasil")
    
    
    def detect_faces(self, frame: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """
        Deteksi wajah dan landmarks dalam frame.
        
        Args:
            frame: Input frame dari webcam (BGR format dari OpenCV)
            
        Returns:
            faces: List berisi informasi setiap wajah yang terdeteksi
                  Format: [{'landmarks': np.array, 'bbox': (x,y,w,h)}, ...]
            annotated_frame: Frame dengan landmark yang digambar (opsional)
        """
        
        # Convert BGR to RGB (MediaPipe butuh RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        
        # Deteksi landmarks
        results = self.face_mesh.process(frame_rgb)
        
        faces = []
        annotated_frame = frame.copy()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks ke format numpy array dengan koordinat pixel
                landmarks = np.array(
                    [(lm.x * width, lm.y * height) for lm in face_landmarks.landmark],
                    dtype=np.float32
                )
                
                # Hitung bounding box dari landmarks
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                
                x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                
                # Tambahkan padding ke bounding box (opsional, untuk lebih bagus)
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(width, x_max + padding)
                y_max = min(height, y_max + padding)
                
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                
                # Simpan info wajah
                faces.append({
                    'landmarks': landmarks,
                    'bbox': bbox,
                    'face_landmarks': face_landmarks
                })
                
                # Gambar bounding box (untuk debug)
                if config.SHOW_FACIAL_LANDMARKS:
                    cv2.rectangle(annotated_frame, 
                                (x_min, y_min), 
                                (x_max, y_max), 
                                config.COLOR_BLUE, 2)
        
        return faces, annotated_frame
    
    
    def get_eye_landmarks(self, landmarks: np.ndarray, 
                         eye: str = "left") -> np.ndarray:
        """
        Ekstrak landmark mata dari 468 landmark face.
        
        Args:
            landmarks: Array berisi 468 facial landmarks
            eye: "left" atau "right"
            
        Returns:
            Eye landmarks dalam format (6, 2) untuk perhitungan EAR
        """
        
        # Gunakan 6 points per mata untuk EAR calculation (lebih cepat)
        # Format: [P1(in), P2(in), P3(up), P4(down), P5(out), P6(out)]
        
        if eye.lower() == "left":
            # Koordinat mata kiri (MediaPipe indexing)
            indices = [362, 385, 387, 263, 373, 380]  # 6 points
        elif eye.lower() == "right":
            # Koordinat mata kanan (MediaPipe indexing)
            indices = [33, 160, 158, 133, 153, 145]  # 6 points
        else:
            raise ValueError("eye harus 'left' atau 'right'")
        
        # Extract eye coordinates
        eye_landmarks = landmarks[indices]
        
        return eye_landmarks
    
    
    def draw_eye_landmarks(self, frame: np.ndarray, 
                           eye_landmarks: np.ndarray,
                           color: Tuple[int, int, int] = config.COLOR_GREEN) -> np.ndarray:
        """
        Gambar eye landmarks pada frame untuk debugging.
        
        Args:
            frame: Input frame
            eye_landmarks: Koordinat eye landmarks
            color: Warna untuk menggambar
            
        Returns:
            Frame dengan eye landmarks yang digambar
        """
        
        frame_copy = frame.copy()
        
        # Gambar points
        for point in eye_landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame_copy, (x, y), 2, color, -1)
        
        # Gambar garis connecting points (forming eye shape)
        for i in range(len(eye_landmarks) - 1):
            pt1 = tuple(map(int, eye_landmarks[i]))
            pt2 = tuple(map(int, eye_landmarks[i + 1]))
            cv2.line(frame_copy, pt1, pt2, color, 1)
        
        # Close the loop
        pt1 = tuple(map(int, eye_landmarks[-1]))
        pt2 = tuple(map(int, eye_landmarks[0]))
        cv2.line(frame_copy, pt1, pt2, color, 1)
        
        return frame_copy
    
    
    def get_face_region(self, frame: np.ndarray, bbox: Tuple) -> np.ndarray:
        """
        Ekstrak region wajah dari frame menggunakan bounding box.
        
        Args:
            frame: Input frame
            bbox: Bounding box format (x, y, width, height)
            
        Returns:
            Cropped face region
        """
        
        x, y, w, h = bbox
        face_region = frame[y:y+h, x:x+w]
        
        return face_region
