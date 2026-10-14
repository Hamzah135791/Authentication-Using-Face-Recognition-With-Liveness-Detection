# ===============================================
# UTILITY SCRIPT: CREATE FACE DATABASE
# Fungsi: Membuat/mengumpulkan database wajah dari webcam
# ===============================================
# 
# Penggunaan:
# python create_face_database.py
# 
# Instruksi:
# 1. Input nama orang
# 2. Tekan 's' untuk save foto wajah
# 3. Ambil beberapa pose berbeda (depan, kiri, kanan)
# 4. Tekan 'c' untuk complete dan switch ke orang berikutnya
# 5. Tekan 'q' untuk quit
#
# ===============================================

import cv2
import os
import sys
import config
from modules import FaceDetector, log_message


class FaceDatabaseCreator:
    """Script untuk membuat database wajah dari webcam."""
    
    def __init__(self):
        """Inisialisasi database creator."""
        
        print("\n" + "="*60)
        print("FACE DATABASE CREATOR")
        print("="*60 + "\n")
        
        # Create database folder jika belum ada
        if not os.path.exists(config.FACE_DATABASE_PATH):
            os.makedirs(config.FACE_DATABASE_PATH)
            log_message(f"Created directory: {config.FACE_DATABASE_PATH}", "INFO")
        
        self.face_detector = FaceDetector()
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        
        self.person_name = ""
        self.photo_count = 0
    
    
    def run(self):
        """Jalankan database creator loop."""
        
        while True:
            print("\n" + "="*60)
            person_name = input("Masukkan nama person (atau 'q' untuk quit): ").strip()
            
            if person_name.lower() == 'q':
                log_message("Database creation selesai", "INFO")
                break
            
            if not person_name:
                log_message("Nama tidak boleh kosong", "WARNING")
                continue
            
            self.person_name = person_name
            self.photo_count = 0
            
            print(f"\nCapturing photos untuk: {self.person_name}")
            print("Kontrol:")
            print("  's': Save foto")
            print("  'c': Selesai dengan person ini")
            print("  'q': Quit")
            print("\nSilakan positioning wajah ke depan kamera...\n")
            
            self._capture_photos()
    
    
    def _capture_photos(self):
        """Capture photos untuk satu person."""
        
        while True:
            ret, frame = self.camera.read()
            
            if not ret:
                log_message("Gagal membaca frame dari webcam", "ERROR")
                break
            
            # Detect faces
            faces, annotated_frame = self.face_detector.detect_faces(frame)
            
            # Display info
            info_text = f"Person: {self.person_name} | Photos: {self.photo_count}"
            annotated_frame = cv2.putText(
                annotated_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )
            
            if len(faces) == 0:
                notice = "Tidak ada wajah terdeteksi"
                annotated_frame = cv2.putText(
                    annotated_frame, notice, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2
                )
            else:
                notice = f"{len(faces)} wajah terdeteksi - Tekan 's' untuk save"
                annotated_frame = cv2.putText(
                    annotated_frame, notice, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )
            
            # Display frame
            cv2.imshow(f"Database Creator - {self.person_name}", annotated_frame)
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                cv2.destroyAllWindows()
                self.camera.release()
                sys.exit(0)
            
            elif key == ord('s') and len(faces) > 0:
                # Save foto
                self.photo_count += 1
                
                # Generate filename
                if self.photo_count == 1:
                    filename = f"{self.person_name}.jpg"
                else:
                    filename = f"{self.person_name}_{self.photo_count}.jpg"
                
                filepath = os.path.join(config.FACE_DATABASE_PATH, filename)
                
                # Save foto
                cv2.imwrite(filepath, frame)
                log_message(f"Saved: {filename}", "INFO")
                
                # Visual feedback
                print(f"✓ Foto ke-{self.photo_count} disimpan")
            
            elif key == ord('c'):
                # Complete person
                if self.photo_count == 0:
                    log_message("Tidak ada foto yang disimpan untuk person ini", "WARNING")
                else:
                    log_message(f"Completed: {self.person_name} ({self.photo_count} photos)", "INFO")
                
                cv2.destroyAllWindows()
                break


def main():
    """Entry point."""
    
    creator = FaceDatabaseCreator()
    
    try:
        creator.run()
    except KeyboardInterrupt:
        log_message("Program dihentikan", "INFO")
    finally:
        creator.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()