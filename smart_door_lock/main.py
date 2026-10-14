# ===============================================
# SMART DOOR LOCK - LIVENESS DETECTION SYSTEM
# Judul Skripsi: "Deteksi Liveness Berbasis Kedipan Mata 
#                 untuk Autentikasi Wajah pada Smart Door Lock Berbasis Raspberry Pi"
# ===============================================
# 
# ALUR SISTEM:
# 1. Buka webcam dan capture frame
# 2. Deteksi wajah menggunakan MediaPipe Face Mesh
# 3. Ekstrak landmark mata kiri dan kanan
# 4. Hitung Eye Aspect Ratio (EAR) untuk deteksi blink
# 5. JIKA liveness terdeteksi (ada blink):
#    - Jalankan face recognition terhadap database
#    - Jika dikenali: PINTU TERBUKA
#    - Jika tidak dikenali: AKSES DITOLAK
# 6. JIKA liveness belum terdeteksi:
#    - Tampilkan pesan "Silakan Berkedip"
# 
# ===============================================

import cv2
import numpy as np
import sys
import os
from typing import Optional, Tuple

# Import custom modules
import config
from modules import (
    FaceDetector, 
    LivenessDetector, 
    FaceRecognizer,
    FPSCounter,
    DisplayHelper,
    log_message
)


class SmartDoorLockSystem:
    """
    Main class untuk Smart Door Lock System dengan Liveness Detection.
    Mengintegrasikan Face Detection, Liveness Detection, dan Face Recognition.
    """
    
    def __init__(self):
        """Inisialisasi semua komponen sistem."""
        
        print("\n" + "="*60)
        print("SMART DOOR LOCK - LIVENESS DETECTION SYSTEM")
        print("="*60 + "\n")
        
        # Initialize modules
        log_message("Inisialisasi Face Detector...", "INFO")
        self.face_detector = FaceDetector()
        
        log_message("Inisialisasi Liveness Detector...", "INFO")
        self.liveness_detector = LivenessDetector()
        
        log_message("Inisialisasi Face Recognizer...", "INFO")
        self.face_recognizer = FaceRecognizer()
        
        # Check database
        if self.face_recognizer.is_database_empty():
            log_message("Database wajah kosong! Harap masukkan foto ke folder 'face_database/'", 
                       "WARNING")
            db_info = self.face_recognizer.get_database_info()
        else:
            db_info = self.face_recognizer.get_database_info()
            log_message(f"Database loaded: {db_info['total_persons']} person(s), "
                       f"{db_info['total_photos']} photo(s)", "INFO")
            for person, count in db_info['person_counts'].items():
                print(f"  - {person}: {count} photo(s)")
        
        # Initialize FPS counter
        self.fps_counter = FPSCounter()
        
        # Initialize webcam
        log_message("Inisialisasi Webcam...", "INFO")
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        if not self.camera.isOpened():
            log_message("Gagal membuka webcam!", "ERROR")
            sys.exit(1)
        
        log_message("Sistem siap! Tekan 'q' untuk quit, 's' untuk save frame, 'r' untuk reset", "INFO")
        print("="*60 + "\n")
    
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process satu frame melalui seluruh pipeline sistem.
        
        Args:
            frame: Input frame dari webcam (BGR format)
            
        Returns:
            Tuple (processed_frame, results_dict)
        """
        
        # Convert BGR to RGB untuk face_recognition dan mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        
        # ========== STEP 1: DETEKSI WAJAH ==========
        faces, face_annotated = self.face_detector.detect_faces(frame)
        
        # Results dictionary
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
        
        # Gunakan frame terakhir sebagai base untuk drawing
        processed_frame = face_annotated.copy()
        
        # ========== STEP 2: PROCESS SETIAP WAJAH ==========
        if len(faces) > 0:
            results['face_detected'] = True
            
            # Gunakan wajah pertama saja (sesuai config MEDIAPIPE_MAX_NUM_FACES = 1)
            face = faces[0]
            landmarks = face['landmarks']
            bbox = face['bbox']
            
            # ========== STEP 3: EKSTRAK LANDMARK MATA DAN HITUNG EAR ==========
            # Mata kiri
            left_eye_landmarks = self.face_detector.get_eye_landmarks(landmarks, "left")
            ear_left = self.liveness_detector.calculate_ear(left_eye_landmarks)
            
            # Mata kanan
            right_eye_landmarks = self.face_detector.get_eye_landmarks(landmarks, "right")
            ear_right = self.liveness_detector.calculate_ear(right_eye_landmarks)
            
            # Simpan ke results
            results['ear_left'] = ear_left
            results['ear_right'] = ear_right
            
            # ========== STEP 4: DETEKSI BLINK (LIVENESS) ==========
            blink_detected = self.liveness_detector.detect_blink(ear_left, ear_right)
            self.liveness_detector.update_liveness(blink_detected)
            
            liveness_status = self.liveness_detector.liveness_status
            results['liveness_status'] = liveness_status
            results['avg_ear'] = self.liveness_detector.get_average_ear()
            
            # ========== STEP 5: FACE RECOGNITION (JIKA LIVENESS DETECTED) ==========
            if liveness_status:
                # Gunakan frame RGB untuk face_recognition library
                face_location = (bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0])
                face_encoding = self.face_recognizer.encode_face(frame_rgb, face_location)
                
                if face_encoding is not None:
                    # Recognize wajah
                    person_name, confidence = self.face_recognizer.recognize_face(face_encoding)
                    
                    results['recognition_result'] = person_name
                    results['recognition_confidence'] = confidence
                else:
                    results['recognition_result'] = "Face Encoding Error"
            else:
                # Belum ada liveness
                results['recognition_result'] = "Waiting for liveness..."
            
            # ========== STEP 6: DRAW ANNOTATION ==========
            # Draw bounding box
            x, y, w, h = bbox
            processed_frame = cv2.rectangle(processed_frame, (x, y), (x+w, y+h), 
                                           config.COLOR_GREEN, 2)
            
            # Draw eye landmarks jika enabled
            if config.SHOW_FACIAL_LANDMARKS:
                processed_frame = self.face_detector.draw_eye_landmarks(
                    processed_frame, left_eye_landmarks, config.COLOR_YELLOW
                )
                processed_frame = self.face_detector.draw_eye_landmarks(
                    processed_frame, right_eye_landmarks, config.COLOR_YELLOW
                )
        else:
            # Tidak ada wajah terdeteksi
            results['recognition_result'] = "No face detected"
        
        # ========== STEP 7: DISPLAY INFORMASI ==========
        processed_frame = self._draw_information(processed_frame, results)
        
        return processed_frame, results
    
    
    def _draw_information(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """
        Draw informasi sistem pada frame.
        
        Args:
            frame: Input frame
            results: Dictionary hasil processing
            
        Returns:
            Frame dengan informasi yang digambar
        """
        
        frame_display = frame.copy()
        
        # ===== Draw EAR Values =====
        if config.SHOW_EAR_VALUES and results['face_detected']:
            ear_text = f"EAR L: {results['ear_left']:.3f} | R: {results['ear_right']:.3f}"
            frame_display = DisplayHelper.put_text(
                frame_display, ear_text, (10, 30),
                color=config.COLOR_WHITE,
                font_scale=0.6,
                thickness=1
            )
        
        # ===== Draw Liveness Status =====
        if results['face_detected']:
            liveness_text, liveness_color = self.liveness_detector.get_liveness_text()
            frame_display = DisplayHelper.put_text(
                frame_display, liveness_text, (10, 60),
                color=liveness_color,
                bg_color=config.COLOR_BLACK,
                font_scale=0.7,
                thickness=2
            )
        
        # ===== Draw Recognition Result =====
        if results['face_detected'] and results['liveness_status']:
            person_name = results['recognition_result']
            confidence = results['recognition_confidence']
            
            # Determine message dan color
            if person_name == "Unknown":
                access_text = "AKSES DITOLAK - Wajah Tidak Dikenali"
                access_color = config.COLOR_RED
            else:
                access_text = f"PINTU TERBUKA - Selamat Datang {person_name}!"
                access_color = config.COLOR_GREEN
            
            # Tambahkan confidence
            access_text_with_conf = f"{access_text} ({confidence:.2%})"
            
            # Draw dengan background
            frame_display = DisplayHelper.put_text(
                frame_display, access_text_with_conf, (10, frame_display.shape[0] - 60),
                color=access_color,
                bg_color=config.COLOR_BLACK,
                font_scale=0.8,
                thickness=2
            )
        
        # ===== Draw Status Panel =====
        status_panel = {
            'Faces': results['face_count'],
            'EAR Avg': f"{results['avg_ear']:.3f}",
            'Threshold': f"{config.EAR_THRESHOLD:.3f}"
        }
        
        # Draw panel di top-right
        panel_x = frame_display.shape[1] - 250
        panel_y = 30
        
        for idx, (label, value) in enumerate(status_panel.items()):
            text = f"{label}: {value}"
            frame_display = DisplayHelper.put_text(
                frame_display, text, (panel_x, panel_y + idx * 30),
                color=config.COLOR_WHITE,
                font_scale=0.6,
                thickness=1
            )
        
        # ===== Draw FPS =====
        if config.SHOW_FPS:
            fps = self.fps_counter.get_fps()
            frame_display = DisplayHelper.draw_fps(frame_display, fps)
        
        return frame_display
    
    
    def run(self):
        """
        Jalankan main loop sistem.
        Tekan 'q' untuk quit, 's' untuk save frame, 'r' untuk reset.
        """
        
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                ret, frame = self.camera.read()
                
                if not ret:
                    log_message("Gagal membaca frame dari webcam", "ERROR")
                    break
                
                # Update FPS counter
                self.fps_counter.update()
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("Smart Door Lock - Liveness Detection", processed_frame)
                
                # ===== KEYBOARD INPUT =====
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Quit
                    log_message("Program dihentikan oleh user", "INFO")
                    break
                
                elif key == ord('s'):
                    # Save current frame
                    frame_count += 1
                    filename = f"debug_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    log_message(f"Frame disimpan: {filename}", "INFO")
                
                elif key == ord('r'):
                    # Reset liveness detection
                    self.liveness_detector.reset()
                    log_message("Liveness detection direset", "INFO")
                
                # Debug print (setiap 30 frame)
                if config.DEBUG_MODE and frame_count % 30 == 0:
                    print(f"\n[Frame {frame_count}]")
                    print(f"  Face Detected: {results['face_detected']}")
                    if results['face_detected']:
                        print(f"  EAR Left: {results['ear_left']:.4f}, Right: {results['ear_right']:.4f}")
                        print(f"  Liveness: {results['liveness_status']}")
                        print(f"  Recognition: {results['recognition_result']} ({results['recognition_confidence']:.2%})")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            log_message("Program diinterupsi", "INFO")
        
        except Exception as e:
            log_message(f"Error: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            self.camera.release()
            cv2.destroyAllWindows()
            log_message("Program selesai", "INFO")


def main():
    """Entry point untuk program."""
    
    # Check if database folder exists
    if not os.path.exists(config.FACE_DATABASE_PATH):
        log_message(f"Database folder '{config.FACE_DATABASE_PATH}' belum ada", "WARNING")
        log_message(f"Membuat folder...", "INFO")
        os.makedirs(config.FACE_DATABASE_PATH)
        log_message(f"Silakan masukkan foto wajah ke folder '{config.FACE_DATABASE_PATH}/'", "INFO")
    
    # Create and run system
    system = SmartDoorLockSystem()
    system.run()


if __name__ == "__main__":
    main()