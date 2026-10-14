import cv2
import face_recognition
import numpy as np
import time
import os
import math

class BlinkDetector:
    """
    Class khusus untuk menangani Liveness Detection (Blink Detection) menggunakan EAR.
    Sangat modular sehingga algoritma Liveness bisa diganti di masa depan..
    """
    def __init__(self, ear_threshold=0.25, consec_frames=2):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.blink_counter = 0

    def hitung_jarak(self, p1, p2):
        """Menghitung jarak Euclidean antara dua titik (x, y)"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def hitung_ear(self, mata):
        """
        Menghitung Eye Aspect Ratio (EAR).
        Format titik mata dlib: 
        p1=kiri, p2=kiri atas, p3=kanan atas, p4=kanan, p5=kanan bawah, p6=kiri bawah
        """
        # Hitung jarak vertikal
        jarak_vertikal_1 = self.hitung_jarak(mata[1], mata[5])
        jarak_vertikal_2 = self.hitung_jarak(mata[2], mata[4])
        
        # Hitung jarak horizontal
        jarak_horizontal = self.hitung_jarak(mata[0], mata[3])
        
        # Rumus EAR
        ear = (jarak_vertikal_1 + jarak_vertikal_2) / (2.0 * jarak_horizontal)
        return ear

    def cek_kedipan(self, left_eye, right_eye):
        """Mendeteksi apakah terjadi kedipan berdasarkan nilai EAR rata-rata"""
        ear_kiri = self.hitung_ear(left_eye)
        ear_kanan = self.hitung_ear(right_eye)
        ear_rata_rata = (ear_kiri + ear_kanan) / 2.0

        is_blinked = False
        
        # Cek apakah mata tertutup (EAR di bawah threshold)
        if ear_rata_rata < self.ear_threshold:
            self.blink_counter += 1
        else:
            # Jika mata kembali terbuka dan frame tertutup mencukupi = Kedipan sukses!
            if self.blink_counter >= self.consec_frames:
                is_blinked = True
            self.blink_counter = 0 # Reset counter
            
        return is_blinked, ear_rata_rata


class FaceAuthSystem:
    """
    Class untuk menangani sistem Face Recognition dari database lokal.
    """
    def __init__(self, dataset_path="dataset_wajah"):
        self.known_face_encodings = []
        self.known_face_names = []
        self._load_dataset(dataset_path)

    def _load_dataset(self, dataset_path):
        """Memuat gambar wajah dari folder dan membuat encodings-nya"""
        print("[INFO] Memuat database wajah...")
        for filename in os.listdir(dataset_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(dataset_path, filename)
                image = face_recognition.load_image_file(filepath)
                
                # Pastikan wajah terdeteksi dalam foto dataset
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    encoding = encodings[0]
                    self.known_face_encodings.append(encoding)
                    name = os.path.splitext(filename)[0] # Ambil nama file tanpa ekstensi
                    self.known_face_names.append(name)
                    print(f"       -> Wajah {name} berhasil dimuat.")
                else:
                    print(f"       -> [WARNING] Wajah tidak ditemukan pada {filename}")

    def recognize_face(self, face_encoding):
        """Mencocokkan wajah dari kamera dengan database"""
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
        name = "Akses Ditolak"

        if True in matches:
            first_match_index = matches.index(True)
            name = self.known_face_names[first_match_index]
            
        return name


def main():
    # Inisialisasi modul
    blink_detector = BlinkDetector(ear_threshold=0.25, consec_frames=2)
    face_auth = FaceAuthSystem(dataset_path="dataset_wajah")
    
    # Buka Webcam
    cap = cv2.VideoCapture(1)
    print("[INFO] Memulai Webcam...")

    # Variabel State System
    liveness_status = False  # False: Belum kedip, True: Asli (Sudah kedip)
    nama_terdeteksi = None

    # Variabel untuk menghitung FPS
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Kamera tidak terdeteksi.")
            break

    # Gunakan ukuran frame asli
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Cari lokasi dan landmark wajah di frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
    
        # Jika tidak ada wajah di frame, reset status liveness
        if len(face_locations) == 0:
            liveness_status = False
            nama_terdeteksi = None

        for (top, right, bottom, left), face_landmarks in zip(face_locations, face_landmarks_list):
            # Kembalikan koordinat wajah ke ukuran asli frame
            # top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

            # 2. Proses Liveness Detection (Cek Kedipan)
            if not liveness_status:
                mata_kiri = face_landmarks['left_eye']
                mata_kanan = face_landmarks['right_eye']
                
                is_blinked, ear_val = blink_detector.cek_kedipan(mata_kiri, mata_kanan)
                
                if is_blinked:
                    liveness_status = True # Wajah terdeteksi asli!
                    print("[INFO] Kedipan terdeteksi! Liveness = Asli.")
                
                # Visualisasi instruksi kedip
                cv2.putText(frame, "SILAKAN BERKEDIP", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 2)
                cv2.putText(frame, f"EAR: {ear_val:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 3. Proses Face Recognition (Hanya jika liveness = True)
            else:
                if nama_terdeteksi is None:
                    # Ambil encoding wajah di area yang ditemukan (gunakan rgb_frame dan koordinat asli)
                    face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])

                    if len(face_encodings) > 0:
                        nama_terdeteksi = face_auth.recognize_face(face_encodings[0])

                # Visualisasi hasil autentikasi
                color = (0, 255, 0) if nama_terdeteksi != "Akses Ditolak" else (0, 0, 255)
                teks_status = f"{nama_terdeteksi} - PINTU TERBUKA" if nama_terdeteksi != "Akses Ditolak" else "AKSES DITOLAK"
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, teks_status, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 4. Kalkulasi dan Tampilkan FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Tampilkan ke layar
        cv2.imshow("Smart Door Lock - Prototype", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()