# ===============================================
# ALTERNATIVE MODULE: FACE DETECTOR (OpenCV)
# Fungsi: Deteksi wajah menggunakan OpenCV Cascade Classifier
# Library: OpenCV (built-in cascade)
# ===============================================
# 
# File ini adalah FALLBACK jika MediaPipe gagal
# Gunakan ini jika face_detector.py error
#
# ===============================================

import cv2
import numpy as np
from typing import List, Tuple, Optional
import config


class FaceDetectorOpenCV:
    """
    Alternative Face Detector menggunakan OpenCV Cascade Classifier.
    Lebih ringan tapi kurang akurat dibanding MediaPipe.
    """
    
    def __init__(self):
        """Inisialisasi OpenCV Cascade Classifier untuk face detection."""
        
        # Load pre-trained Haar Cascade untuk face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Optional: Load eye cascade untuk mata
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        print("[FaceDetectorOpenCV] Inisialisasi OpenCV Cascade Classifier berhasil")
        print(f"  - Face Cascade: {cascade_path}")
        print(f"  - Eye Cascade: {eye_cascade_path}")
    
    
    def detect_faces(self, frame: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """
        Deteksi wajah menggunakan OpenCV Cascade Classifier.
        
        Args:
            frame: Input frame dari webcam (BGR format)
            
        Returns:
            Tuple (faces_list, annotated_frame)
        """
        
        # Convert ke grayscale untuk detection (lebih cepat)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram untuk improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Deteksi wajah
        detected_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,          # Sensitivity (semakin kecil semakin sensitif)
            minNeighbors=5,           # Minimum neighbors (semakin tinggi semakin ketat)
            minSize=(30, 30),         # Minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        
        for (x, y, w, h) in detected_faces:
            # Tentukan ROI untuk mata (bagian atas dari wajah)
            roi_gray = gray[y:y+h//2, x:x+w]
            roi_color = frame[y:y+h//2, x:x+w]
            
            # Deteksi mata dalam ROI
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Create dummy landmarks (placeholder untuk compatibility)
            # Format: 468 points seperti MediaPipe (akan di-simplify)
            landmarks = self._create_dummy_landmarks(x, y, w, h, frame.shape)
            
            # Store face info
            faces.append({
                'landmarks': landmarks,
                'bbox': (x, y, w, h),
                'eyes_detected': len(eyes),
                'roi_eyes': eyes
            })
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), 
                         config.COLOR_BLUE, 2)
            
            # Draw eyes jika terdeteksi
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), 
                             config.COLOR_GREEN, 2)
        
        return faces, annotated_frame
    
    
    def _create_dummy_landmarks(self, x: int, y: int, w: int, h: int, 
                               frame_shape: Tuple) -> np.ndarray:
        """
        Create dummy landmarks untuk compatibility dengan liveness_detector.
        OpenCV tidak provide detailed landmarks seperti MediaPipe,
        jadi kita create synthetic ones untuk eye region.
        
        Args:
            x, y, w, h: Face bounding box coordinates
            frame_shape: Shape dari frame (h, w, c)
            
        Returns:
            Array landmarks (468, 2)
        """
        
        # Initialize empty landmarks array
        landmarks = np.zeros((468, 2), dtype=np.float32)
        
        # Approximate eye positions (simplified)
        # Left eye (approximate center)
        left_eye_x = x + w // 3
        left_eye_y = y + h // 3
        
        # Right eye (approximate center)
        right_eye_x = x + 2 * w // 3
        right_eye_y = y + h // 3
        
        # For compatibility dengan liveness_detector, 
        # set key eye landmark indices
        
        # Mata kiri (6 points untuk EAR calculation)
        eye_w = w // 8
        eye_h = h // 6
        
        # LEFT EYE (indices 362, 385, 387, 263, 373, 380)
        landmarks[362] = [left_eye_x - eye_w, left_eye_y]           # Inner
        landmarks[385] = [left_eye_x - eye_w//2, left_eye_y - eye_h]  # Top left
        landmarks[387] = [left_eye_x + eye_w//2, left_eye_y - eye_h]  # Top right
        landmarks[263] = [left_eye_x + eye_w, left_eye_y]           # Outer
        landmarks[373] = [left_eye_x - eye_w//2, left_eye_y + eye_h]  # Bottom left
        landmarks[380] = [left_eye_x + eye_w//2, left_eye_y + eye_h]  # Bottom right
        
        # RIGHT EYE (indices 33, 160, 158, 133, 153, 145)
        landmarks[33] = [right_eye_x - eye_w, right_eye_y]           # Inner
        landmarks[160] = [right_eye_x - eye_w//2, right_eye_y - eye_h]  # Top left
        landmarks[158] = [right_eye_x + eye_w//2, right_eye_y - eye_h]  # Top right
        landmarks[133] = [right_eye_x + eye_w, right_eye_y]           # Outer
        landmarks[153] = [right_eye_x - eye_w//2, right_eye_y + eye_h]  # Bottom left
        landmarks[145] = [right_eye_x + eye_w//2, right_eye_y + eye_h]  # Bottom right
        
        # Nose landmark (index 1)
        landmarks[1] = [x + w // 2, y + h // 2]
        
        return landmarks
    
    
    def get_eye_landmarks(self, landmarks: np.ndarray, 
                         eye: str = "left") -> np.ndarray:
        """
        Ekstrak landmark mata dari landmarks array.
        
        Args:
            landmarks: Array berisi facial landmarks
            eye: "left" atau "right"
            
        Returns:
            Eye landmarks dalam format (6, 2)
        """
        
        if eye.lower() == "left":
            indices = [362, 385, 387, 263, 373, 380]
        elif eye.lower() == "right":
            indices = [33, 160, 158, 133, 153, 145]
        else:
            raise ValueError("eye harus 'left' atau 'right'")
        
        eye_landmarks = landmarks[indices]
        return eye_landmarks
    
    
    def draw_eye_landmarks(self, frame: np.ndarray, 
                          eye_landmarks: np.ndarray,
                          color: Tuple[int, int, int] = config.COLOR_GREEN) -> np.ndarray:
        """
        Gambar eye landmarks pada frame.
        
        Args:
            frame: Input frame
            eye_landmarks: Koordinat eye landmarks
            color: Warna untuk menggambar
            
        Returns:
            Frame dengan eye landmarks
        """
        
        frame_copy = frame.copy()
        
        # Gambar points
        for point in eye_landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame_copy, (x, y), 2, color, -1)
        
        # Gambar garis connecting points
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
        Ekstrak region wajah dari frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box format (x, y, w, h)
            
        Returns:
            Cropped face region
        """
        
        x, y, w, h = bbox
        face_region = frame[y:y+h, x:x+w]
        
        return face_region


def main():
    """Test function untuk OpenCV Face Detector."""
    
    print("\nTesting OpenCV Face Detector...")
    
    detector = FaceDetectorOpenCV()
    camera = cv2.VideoCapture(0)
    
    print("Tekan 'q' untuk quit\n")
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            faces, annotated = detector.detect_faces(frame)
            
            # Add text
            text = f"Faces: {len(faces)}"
            cv2.putText(annotated, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0), 2)
            
            cv2.imshow("OpenCV Face Detector", annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
