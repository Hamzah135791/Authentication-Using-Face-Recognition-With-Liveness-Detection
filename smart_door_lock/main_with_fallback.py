# ===============================================
# SMART DOOR LOCK - LIVENESS DETECTION SYSTEM
# Version: 2.0 WITH MEDIAPIPE FALLBACK
# ===============================================
# 
# Script ini sama dengan main.py, tapi dengan:
# - Automatic fallback ke OpenCV jika MediaPipe gagal
# - Better error handling
# - Fallback notification
#
# ===============================================

import cv2
import numpy as np
import sys
import os
from typing import Optional, Tuple

# Import custom modules
import config


class SmartDoorLockSystem:
    """
    Main class untuk Smart Door Lock System dengan Liveness Detection.
    Dengan automatic fallback ke OpenCV jika MediaPipe tidak tersedia.
    """
    
    def __init__(self):
        """Inisialisasi semua komponen sistem."""
        
        print("\n" + "="*60)
        print("SMART DOOR LOCK - LIVENESS DETECTION SYSTEM")
        print("="*60 + "\n")
        
        # Try import modules dengan fallback
        self.face_detector = self._init_face_detector()
        self.liveness_detector = self._init_liveness_detector()
        self.face_recognizer = self._init_face_recognizer()
        self.fps_counter = self._init_fps_counter()
        self.display_helper = self._init_display_helper()
        
        # Initialize webcam
        print("[INFO] Inisialisasi Webcam...")
        # self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        self.camera = cv2.VideoCapture(1)

        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        if not self.camera.isOpened():
            print("[ERROR] Gagal membuka webcam!")
            sys.exit(1)
        
        print("[INFO] Sistem siap! Tekan 'q' untuk quit")
        print("="*60 + "\n")
    
    
    def _init_face_detector(self):
        """Initialize Face Detector dengan fallback."""
        
        print("[INFO] Inisialisasi Face Detector...")
        
        try:
            from modules import FaceDetector
            print("  ✓ Mencoba menggunakan MediaPipe...")
            detector = FaceDetector()
            print("  ✓ MediaPipe berhasil!")
            return detector
        
        except AttributeError as e:
            print(f"  ✗ MediaPipe gagal: {str(e)}")
            print("  → Fallback ke OpenCV Cascade Classifier...")
            
            try:
                from modules.face_detector_opencv import FaceDetectorOpenCV
                detector = FaceDetectorOpenCV()
                print("  ✓ OpenCV berhasil!")
                print("\n  ⚠️  WARNING: Menggunakan OpenCV Cascade (fallback)")
                print("      Akurasi lebih rendah dibanding MediaPipe")
                print("      Jalankan: python fix_mediapipe.py untuk fix MediaPipe")
                return detector
            
            except Exception as e2:
                print(f"  ✗ OpenCV juga gagal: {str(e2)}")
                sys.exit(1)
        
        except ImportError as e:
            print(f"  ✗ Import error: {str(e)}")
            print("  → Fallback ke OpenCV Cascade Classifier...")
            
            try:
                from modules.face_detector_opencv import FaceDetectorOpenCV
                detector = FaceDetectorOpenCV()
                print("  ✓ OpenCV berhasil!")
                return detector
            
            except Exception as e2:
                print(f"  ✗ OpenCV juga gagal: {str(e2)}")
                sys.exit(1)
    
    
    def _init_liveness_detector(self):
        """Initialize Liveness Detector."""
        
        print("[INFO] Inisialisasi Liveness Detector...")
        
        try:
            from modules import LivenessDetector
            detector = LivenessDetector()
            return detector
        
        except Exception as e:
            print(f"[ERROR] Liveness Detector gagal: {str(e)}")
            sys.exit(1)
    
    
    def _init_face_recognizer(self):
        """Initialize Face Recognizer."""
        
        print("[INFO] Inisialisasi Face Recognizer...")
        
        try:
            from modules import FaceRecognizer
            recognizer = FaceRecognizer()
            
            db_info = recognizer.get_database_info()
            
            if db_info['total_persons'] > 0:
                print(f"  ✓ Database loaded: {db_info['total_persons']} person(s), "
                      f"{db_info['total_photos']} photo(s)")
                for person, count in db_info['person_counts'].items():
                    print(f"    - {person}: {count} photo(s)")
            else:
                print("  ⚠️  Database kosong! Silakan masukkan foto ke folder 'face_database/'")
                print("      Gunakan: python create_face_database.py")
            
            return recognizer
        
        except Exception as e:
            print(f"[ERROR] Face Recognizer gagal: {str(e)}")
            sys.exit(1)
    
    
    def _init_fps_counter(self):
        """Initialize FPS Counter."""
        
        try:
            from modules import FPSCounter
            return FPSCounter()
        except Exception as e:
            print(f"[WARNING] FPS Counter gagal: {str(e)}")
            return None
    
    
    def _init_display_helper(self):
        """Initialize Display Helper."""
        
        try:
            from modules import DisplayHelper
            return DisplayHelper
        except Exception as e:
            print(f"[WARNING] Display Helper gagal: {str(e)}")
            return None
    
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Process satu frame melalui pipeline sistem."""
        
        # Detect faces
        faces, face_annotated = self.face_detector.detect_faces(frame)
        
        results = {
            'face_detected': False,
            'face_count': len(faces),
            'liveness_status': False,
            'recognition_result': "Waiting for face detection...",
            'recognition_confidence': 0.0,
            'ear_left': 0.0,
            'ear_right': 0.0,
            'avg_ear': 0.0
        }
        
        processed_frame = face_annotated.copy()
        
        if len(faces) > 0:
            results['face_detected'] = True
            face = faces[0]
            landmarks = face['landmarks']
            bbox = face['bbox']
            
            # Get eye landmarks
            left_eye_landmarks = self.face_detector.get_eye_landmarks(landmarks, "left")
            right_eye_landmarks = self.face_detector.get_eye_landmarks(landmarks, "right")
            
            # Calculate EAR
            ear_left = self.liveness_detector.calculate_ear(left_eye_landmarks)
            ear_right = self.liveness_detector.calculate_ear(right_eye_landmarks)
            
            results['ear_left'] = ear_left
            results['ear_right'] = ear_right
            
            # Detect blink
            blink_detected = self.liveness_detector.detect_blink(ear_left, ear_right)
            self.liveness_detector.update_liveness(blink_detected)
            
            liveness_status = self.liveness_detector.liveness_status
            results['liveness_status'] = liveness_status
            results['avg_ear'] = self.liveness_detector.get_average_ear()
            
            # Face recognition jika liveness detected
            if liveness_status:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_location = (bbox[1], bbox[0] + bbox[2], 
                                    bbox[1] + bbox[3], bbox[0])
                    face_encoding = self.face_recognizer.encode_face(frame_rgb, face_location)
                    
                    if face_encoding is not None:
                        person_name, confidence = self.face_recognizer.recognize_face(face_encoding)
                        results['recognition_result'] = person_name
                        results['recognition_confidence'] = confidence
                    else:
                        results['recognition_result'] = "Face Encoding Error"
                
                except Exception as e:
                    results['recognition_result'] = f"Recognition Error: {str(e)[:30]}"
            else:
                results['recognition_result'] = "Waiting for liveness..."
            
            # Draw landmarks
            processed_frame = self.face_detector.draw_eye_landmarks(
                processed_frame, left_eye_landmarks, config.COLOR_YELLOW
            )
            processed_frame = self.face_detector.draw_eye_landmarks(
                processed_frame, right_eye_landmarks, config.COLOR_YELLOW
            )
        
        # Draw information
        processed_frame = self._draw_information(processed_frame, results)
        
        return processed_frame, results
    
    
    def _draw_information(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw informasi pada frame."""
        
        frame_display = frame.copy()
        
        # Draw EAR values
        if config.SHOW_EAR_VALUES and results['face_detected']:
            ear_text = f"EAR L: {results['ear_left']:.3f} | R: {results['ear_right']:.3f}"
            frame_display = cv2.putText(
                frame_display, ear_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                config.COLOR_WHITE, 1
            )
        
        # Draw Liveness Status
        if results['face_detected']:
            liveness_text, liveness_color = self.liveness_detector.get_liveness_text()
            frame_display = cv2.putText(
                frame_display, liveness_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                liveness_color, 2
            )
        
        # Draw Recognition Result
        if results['face_detected'] and results['liveness_status']:
            person_name = results['recognition_result']
            confidence = results['recognition_confidence']
            
            if person_name == "Unknown":
                access_text = "AKSES DITOLAK - Wajah Tidak Dikenali"
                access_color = config.COLOR_RED
            else:
                access_text = f"PINTU TERBUKA - Selamat Datang {person_name}!"
                access_color = config.COLOR_GREEN
            
            access_text_with_conf = f"{access_text} ({confidence:.2%})"
            
            frame_display = cv2.putText(
                frame_display, access_text_with_conf, 
                (10, frame_display.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                access_color, 2
            )
        
        # Draw FPS
        if config.SHOW_FPS and self.fps_counter is not None:
            fps = self.fps_counter.get_fps()
            fps_text = f"FPS: {fps:.2f}"
            frame_display = cv2.putText(
                frame_display, fps_text,
                (frame_display.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                config.COLOR_GREEN, 2
            )
        
        return frame_display
    
    
    def run(self):
        """Jalankan main loop sistem."""
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.camera.read()
                
                if not ret:
                    print("[ERROR] Gagal membaca frame dari webcam")
                    break
                
                # Update FPS
                if self.fps_counter is not None:
                    self.fps_counter.update()
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Display
                cv2.imshow("Smart Door Lock - Liveness Detection", processed_frame)
                
                # Keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("[INFO] Program dihentikan oleh user")
                    break
                
                elif key == ord('s'):
                    frame_count += 1
                    filename = f"debug_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"[INFO] Frame disimpan: {filename}")
                
                elif key == ord('r'):
                    self.liveness_detector.reset()
                    print("[INFO] Liveness detection direset")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("[INFO] Program diinterupsi")
        
        except Exception as e:
            print(f"[ERROR] Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            print("[INFO] Program selesai")


def main():
    """Entry point."""
    
    # Check database folder
    if not os.path.exists(config.FACE_DATABASE_PATH):
        print(f"[INFO] Database folder '{config.FACE_DATABASE_PATH}' belum ada")
        print(f"[INFO] Membuat folder...")
        os.makedirs(config.FACE_DATABASE_PATH)
        print(f"[INFO] Silakan masukkan foto wajah ke folder '{config.FACE_DATABASE_PATH}/'")
    
    # Run system
    system = SmartDoorLockSystem()
    system.run()


if __name__ == "__main__":
    main()
