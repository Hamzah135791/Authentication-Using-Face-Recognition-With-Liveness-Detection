# ===============================================
# MODULE: LIVENESS DETECTOR (BLINK DETECTION)
# Fungsi: Mendeteksi kedipan mata sebagai anti-spoofing
# Method: Eye Aspect Ratio (EAR)
# ===============================================

import numpy as np
from typing import Tuple, Optional
from collections import deque
import config


class LivenessDetector:
    """
    Class untuk mendeteksi liveness (keaslian) wajah melalui blink detection.
    Menggunakan Eye Aspect Ratio (EAR) untuk mendeteksi pembukaan/penutupan mata.
    
    EAR = ||P2-P6|| + ||P3-P5|| / (2 * ||P1-P4||)
    
    Dimana:
    - P1: Inner corner (sudut dalam mata)
    - P2, P3: Top eye lid (kelopak atas)
    - P4: Outer corner (sudut luar mata)
    - P5, P6: Bottom eye lid (kelopak bawah)
    """
    
    def __init__(self):
        """Inisialisasi Liveness Detector."""
        self.liveness_status = False  # Status liveness: True/False
        self.blink_count = 0           # Counter jumlah blink terdeteksi
        self.frame_counter = 0         # Counter frame dengan EAR rendah (blink)
        self.ear_history = deque(maxlen=30)  # Histori EAR untuk smoothing
        
        if config.DEBUG_MODE:
            print("[LivenessDetector] Inisialisasi berhasil")
    
    
    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Hitung Eye Aspect Ratio (EAR) dari 6 point eye landmarks.
        
        Formula EAR:
        EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
        
        Args:
            eye_landmarks: Array shape (6, 2) berisi koordinat mata
                          Format: [P1_in, P2_top_left, P3_top_right, 
                                   P4_out, P5_bot_left, P6_bot_right]
                          
        Returns:
            Nilai EAR (float)
        """
        
        # Extract coordinates
        # P1: inner corner (sudut dalam)
        # P2, P3: top eyelid (kelopak atas)
        # P4: outer corner (sudut luar)
        # P5, P6: bottom eyelid (kelopak bawah)
        
        P1 = eye_landmarks[0]  # inner corner
        P2 = eye_landmarks[1]  # top left
        P3 = eye_landmarks[2]  # top right
        P4 = eye_landmarks[3]  # outer corner
        P5 = eye_landmarks[4]  # bottom left
        P6 = eye_landmarks[5]  # bottom right
        
        # Hitung jarak Euclidean
        # ||P2-P6||: vertikal distance antara kelopak atas-bawah (kiri)
        dist_P2_P6 = np.linalg.norm(P2 - P6)
        
        # ||P3-P5||: vertikal distance antara kelopak atas-bawah (kanan)
        dist_P3_P5 = np.linalg.norm(P3 - P5)
        
        # ||P1-P4||: horizontal distance antara sudut dalam-luar
        dist_P1_P4 = np.linalg.norm(P1 - P4)
        
        # Hitung EAR
        # Jika pembilang kecil (mata tertutup): EAR kecil
        # Jika pembilang besar (mata terbuka): EAR besar
        if dist_P1_P4 == 0:  # Proteksi division by zero
            ear = 0
        else:
            ear = (dist_P2_P6 + dist_P3_P5) / (2.0 * dist_P1_P4)
        
        return ear
    
    
    def detect_blink(self, ear_left: float, ear_right: float) -> bool:
        """
        Deteksi blink berdasarkan EAR dari kedua mata.
        
        Args:
            ear_left: Eye Aspect Ratio mata kiri
            ear_right: Eye Aspect Ratio mata kanan
            
        Returns:
            True jika terdeteksi blink, False sebaliknya
        """
        
        # Gunakan method yang dikonfigurasi
        if config.EAR_CALCULATION_METHOD == "left_eye":
            ear = ear_left
        elif config.EAR_CALCULATION_METHOD == "right_eye":
            ear = ear_right
        else:  # "both_eyes" - rata-rata kedua mata
            ear = (ear_left + ear_right) / 2.0
        
        # Simpan ke history
        self.ear_history.append(ear)
        
        # Deteksi blink: jika EAR < threshold
        is_blinking = ear < config.EAR_THRESHOLD
        
        if is_blinking:
            # Increment frame counter jika mata tertutup
            self.frame_counter += 1
        else:
            # Reset frame counter jika mata terbuka
            # Tapi cek dulu apakah sudah memenuhi konsekutif frames
            if self.frame_counter >= config.CONSECUTIVE_FRAMES_BLINK:
                # Blink terdeteksi!
                self.blink_count += 1
                self.frame_counter = 0
                return True
            else:
                # Mata terbuka, reset counter
                self.frame_counter = 0
                return False
        
        return False
    
    
    def update_liveness(self, blink_detected: bool) -> bool:
        """
        Update status liveness berdasarkan blink yang terdeteksi.
        
        Args:
            blink_detected: Apakah blink baru terdeteksi
            
        Returns:
            Status liveness saat ini (True = asli, False = belum asli)
        """
        
        if blink_detected:
            if self.blink_count >= config.BLINKS_REQUIRED:
                self.liveness_status = True
        
        return self.liveness_status
    
    
    def get_average_ear(self) -> float:
        """
        Dapatkan rata-rata EAR dari history (untuk smoothing).
        
        Returns:
            Rata-rata EAR dari 30 frame terakhir
        """
        
        if len(self.ear_history) == 0:
            return 0.0
        
        return float(np.mean(self.ear_history))
    
    
    def reset(self):
        """Reset semua counter dan status liveness."""
        self.liveness_status = False
        self.blink_count = 0
        self.frame_counter = 0
        self.ear_history.clear()
        
        if config.DEBUG_MODE:
            print("[LivenessDetector] Reset status")
    
    
    def get_status(self) -> dict:
        """
        Dapatkan status liveness detection dalam format dict.
        
        Returns:
            Dictionary berisi:
            - 'is_alive': Status liveness (bool)
            - 'blink_count': Jumlah blink terdeteksi (int)
            - 'blink_counter': Frame counter untuk blink current (int)
            - 'avg_ear': Rata-rata EAR (float)
        """
        
        return {
            'is_alive': self.liveness_status,
            'blink_count': self.blink_count,
            'frame_counter': self.frame_counter,
            'avg_ear': self.get_average_ear()
        }
    
    
    def get_liveness_text(self) -> Tuple[str, Tuple[int, int, int]]:
        """
        Dapatkan text dan warna untuk display status liveness.
        
        Returns:
            Tuple (text, color) dalam format BGR
        """
        
        if self.liveness_status:
            return ("Liveness: ASLI (Real)", config.COLOR_GREEN)
        else:
            return (f"Liveness: Silakan Berkedip ({self.blink_count}/{config.BLINKS_REQUIRED})", 
                   config.COLOR_YELLOW)
