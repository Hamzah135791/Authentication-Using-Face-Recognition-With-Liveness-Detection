"""
================================================================================
MODUL 4: METRICS MONITOR, CSV LOGGER, & ON-SCREEN DISPLAY (OSD)
================================================================================
Modul ini bertugas mencatat dan memvisualisasikan parameter performa sistem
untuk kebutuhan analisis kuantitatif pada Skripsi / Tugas Akhir:

Metrik yang Dihitung & Dicatat:
1. FPS (Frames Per Second) real-time dengan exponential moving average.
2. Latensi Inferensi (Inference Time) Landmark & Face Recognition dalam milidetik.
3. Total Latensi Pipeline (End-to-End Latency) dalam milidetik.
4. Penggunaan CPU (%) proses & sistem menggunakan psutil.
5. Penggunaan Memori RAM (MB) proses menggunakan psutil.
6. Riwayat nilai EAR & status transisi liveness.
"""

import os
import csv
import time
import datetime
import psutil
import numpy as np
import cv2


class PerformanceMonitor:
    """
    Kelas pemantau performa sistem (FPS, Latensi, CPU %, RAM MB)
    dan pencatat log CSV untuk pengujian skripsi.
    """
    def __init__(self, log_to_csv=False, log_dir="logs", exp_name="skripsi_run"):
        self.log_to_csv = log_to_csv
        self.log_dir = log_dir
        self.exp_name = exp_name

        # Inisialisasi psutil process
        self.process = psutil.Process(os.getpid())
        # Panggil cpu_percent sekali untuk inisialisasi baseline psutil
        try:
            self.process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        # Variabel FPS & Waktu
        self.fps = 0.0
        self.prev_time = time.perf_counter()
        self.frame_count = 0

        # Cache metrik hardware agar tidak membebani loop per frame
        self.last_hw_sample_time = 0.0
        self.hw_sample_interval = 0.3  # update CPU & RAM tiap 300 ms
        self.cached_cpu_pct = 0.0
        self.cached_ram_mb = 0.0
        self.cached_sys_ram_pct = 0.0

        # File CSV handler
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None

        if self.log_to_csv:
            self.start_csv_logging()

    def start_csv_logging(self):
        """Membuka file CSV baru untuk menyimpan metrik per frame."""
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.log_dir, f"metrics_{self.exp_name}_{timestamp_str}.csv")

        self.csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)

        # Header Kolom Metrik Skripsi
        headers = [
            "Timestamp",
            "Frame_ID",
            "FPS",
            "Landmark_Time_ms",
            "Recog_Time_ms",
            "Total_Pipeline_ms",
            "Process_CPU_Percent",
            "Process_RAM_MB",
            "System_RAM_Percent",
            "EAR_Value",
            "Consecutive_Closed",
            "Blink_Count",
            "Liveness_Status",
            "Recognized_Name",
            "Distance_Score"
        ]
        self.csv_writer.writerow(headers)
        self.csv_file.flush()
        self.log_to_csv = True
        print(f"[INFO] Logging metrik skripsi aktif: {self.csv_path}")

    def toggle_csv_logging(self):
        """Menyalakan / mematikan pencatatan CSV secara interaktif."""
        if self.log_to_csv:
            self.stop_csv_logging()
            return False
        else:
            self.start_csv_logging()
            return True

    def update_hardware_metrics(self):
        """Memperbarui data beban CPU dan konsumsi RAM dengan caching efisien."""
        now = time.perf_counter()
        if (now - self.last_hw_sample_time) >= self.hw_sample_interval:
            self.last_hw_sample_time = now
            try:
                # CPU proses dalam persentase
                self.cached_cpu_pct = self.process.cpu_percent(interval=None)
                # Konsumsi memori RAM (RSS - Resident Set Size) dalam MB
                mem_info = self.process.memory_info()
                self.cached_ram_mb = mem_info.rss / (1024.0 * 1024.0)
                # Persentase total RAM sistem Raspberry Pi
                self.cached_sys_ram_pct = psutil.virtual_memory().percent
            except Exception:
                pass

    def tick(self):
        """Dipanggil di awal setiap iterasi frame untuk menghitung FPS."""
        self.frame_count += 1
        now = time.perf_counter()
        dt = now - self.prev_time
        self.prev_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            if self.fps == 0.0:
                self.fps = instant_fps
            else:
                # Exponential smoothing filter
                self.fps = 0.85 * self.fps + 0.15 * instant_fps

        self.update_hardware_metrics()
        return self.fps

    def log_frame(self, landmark_ms, recog_ms, total_ms, ear_val,
                  consecutive_closed, blink_count, liveness_status,
                  recog_name, distance_score):
        """Mencatat metrik frame ke dalam file CSV jika logging aktif."""
        if not self.log_to_csv or self.csv_writer is None:
            return

        now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = [
            now_iso,
            self.frame_count,
            round(self.fps, 2),
            round(landmark_ms, 2),
            round(recog_ms, 2),
            round(total_ms, 2),
            round(self.cached_cpu_pct, 1),
            round(self.cached_ram_mb, 1),
            round(self.cached_sys_ram_pct, 1),
            round(ear_val, 4),
            consecutive_closed,
            blink_count,
            liveness_status,
            recog_name,
            round(distance_score, 4) if distance_score is not None else -1.0
        ]
        self.csv_writer.writerow(row)

        # Flush tiap 30 frame agar data tersimpan aman tanpa mengorbankan performa disk SD card
        if self.frame_count % 30 == 0:
            self.csv_file.flush()

    def stop_csv_logging(self):
        """Menutup file log CSV dengan aman."""
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            self.log_to_csv = False
            print(f"[INFO] File log metrik skripsi ditutup: {self.csv_path}")

    def close(self):
        self.stop_csv_logging()


class OSDRenderer:
    """
    Kelas perender grafis antarmuka On-Screen Display (OSD / HUD).
    Menampilkan status keamanan, grafik bukaan EAR, dan metrik sistem secara elegan.
    """
    def __init__(self, display_scale=1.0):
        self.display_scale = display_scale

    def draw_ear_bar(self, img, ear_val, threshold, x=10, y=215, w=100, h=12):
        """Menggambar visual progress bar nilai EAR dan garis batas threshold."""
        max_ear = 0.40
        clamped_ear = min(max(ear_val, 0.0), max_ear)
        fill_w = int((clamped_ear / max_ear) * w)

        # Warna bar: Hijau jika terbuka, Merah jika terpejam di bawah threshold
        bar_color = (0, 220, 0) if ear_val >= threshold else (0, 0, 240)

        # Background gelap
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
        # Bar isi
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), bar_color, -1)
        # Border
        cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 180), 1)

        # Garis batas threshold (kuning)
        thresh_x = x + int((threshold / max_ear) * w)
        cv2.line(img, (thresh_x, y - 2), (thresh_x, y + h + 2), (0, 240, 255), 2)

        # Teks nilai EAR
        text = f"EAR: {ear_val:.2f}"
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1)

    def draw_hud(self, img, monitor: PerformanceMonitor, landmark_res,
                 liveness_state_name, status_text, status_subtext,
                 status_color, ear_threshold, required_blinks,
                 current_blinks, sample_count, active_recognizer_name):
        """
        Menggambar keseluruhan antarmuka OSD (Status bar, Bounding Box,
        Landmarks, EAR Gauge, dan Info Metrik Perangkat).
        """
        h, w = img.shape[:2]

        # 1. Gambar Bounding Box Wajah & Titik Landmark Mata
        if landmark_res and landmark_res.face_detected and landmark_res.bbox:
            x1, y1, x2, y2 = landmark_res.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), status_color, 2)

            # Titik-titik landmark mata (Cyan kecil)
            for pt in landmark_res.left_eye + landmark_res.right_eye:
                cv2.circle(img, pt, 2, (255, 255, 0), -1)

        # 2. Header Panel Status di Bagian Atas
        banner_h = 36
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        # Garis aksen status di bawah banner
        cv2.rectangle(img, (0, banner_h - 2), (w, banner_h), status_color, -1)

        cv2.putText(img, status_text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)
        cv2.putText(img, status_subtext, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

        # 3. Bar Visual EAR di Kiri Bawah
        ear_val = landmark_res.ear_avg if landmark_res else 0.0
        self.draw_ear_bar(img, ear_val, ear_threshold, x=8, y=h - 22, w=95, h=12)

        # 4. Kotak Metrik Performa Perangkat (Top-Right HUD)
        hud_w = 98
        hud_h = 58
        hud_x = w - hud_w - 4
        hud_y = 40
        overlay_hud = img.copy()
        cv2.rectangle(overlay_hud, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay_hud, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (80, 80, 80), 1)

        # Nilai metrik
        infer_total_ms = (landmark_res.inference_time_ms if landmark_res else 0.0)
        cv2.putText(img, f"FPS: {monitor.fps:.1f}", (hud_x + 5, hud_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Inf: {infer_total_ms:.0f} ms", (hud_x + 5, hud_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"CPU: {monitor.cached_cpu_pct:.0f}%", (hud_x + 5, hud_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"RAM: {monitor.cached_ram_mb:.0f}MB", (hud_x + 5, hud_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1, cv2.LINE_AA)

        # Indikator status REC CSV jika sedang merekam
        if monitor.log_to_csv:
            cv2.circle(img, (hud_x + hud_w - 10, hud_y + 10), 3, (0, 0, 255), -1)

        # 5. Informasi Status Bawah (Footer)
        footer_text = f"Kedip: {current_blinks}/{required_blinks} | DB: {sample_count} | {active_recognizer_name.upper()}"
        cv2.putText(img, footer_text, (112, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

        # Jika display_scale > 1.0 (misal 2.0 untuk memperbesar ke 640x480 saat sidang skripsi)
        if self.display_scale > 1.0:
            target_w = int(w * self.display_scale)
            target_h = int(h * self.display_scale)
            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        return img
