"""
================================================================================
SISTEM AUTENTIKASI PENGENALAN WAJAH DENGAN LIVENESS DETECTION (EAR)
DIOPTIMALKAN UNTUK RASPBERRY PI 3 (RAM 1 GB, QUAD-CORE ARM CPU)
================================================================================
Tugas Akhir / Skripsi: Implementasi Sistem Autentikasi Biometrik Wajah Real-Time
berbasis Anti-Spoofing Kedipan Mata (Eye Aspect Ratio).

Alur Kerja Pipeline (Liveness First):
1. Background Thread Camera: Mengambil frame 320x240 secara non-blocking.
2. Deteksi Landmark Wajah: Ekstraksi kontur mata menggunakan MediaPipe Face Mesh.
3. Evaluasi Liveness (EAR):
   - Uji bukaan mata menggunakan formula ilmiah Soukupová & Čech (2016).
   - Menghitung transisi kedipan konsekutif (2-3 frame terpejam -> melek).
   - Status Liveness harus "PASSED" sebelum pengenalan wajah dijalankan.
4. Pengenalan Wajah (HANYA jika lolos liveness):
   - Menggunakan LBPH (OpenCV) / MobileFaceNet (TFLite) / Dlib (128D ResNet).
5. Logging Metrik Skripsi:
   - Menghitung FPS, latensi inferensi (ms), penggunaan CPU (%), dan RAM (MB).
   - Menampilkan visual HUD On-Screen Display (OSD) dan ekspor log ke file CSV.
================================================================================
"""

import os
import sys
import time
import argparse

import config

# Definisi State Machine Liveness-First
STATE_NO_FACE = "NO_FACE"
STATE_LIVENESS_CHECKING = "LIVENESS_CHECKING"
STATE_LIVENESS_PASSED = "LIVENESS_PASSED"
STATE_ACCESS_GRANTED = "ACCESS_GRANTED"
STATE_ACCESS_DENIED = "ACCESS_DENIED"


def parse_arguments():
    """Menguraikan parameter baris perintah (CLI arguments)."""
    parser = argparse.ArgumentParser(
        description="Sistem Pengenalan Wajah & Liveness Detection Teroptimasi Raspberry Pi 3"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="mediapipe",
        choices=["mediapipe", "dlib"],
        help="Model deteksi landmark: 'mediapipe' (default) atau 'dlib'"
    )
    parser.add_argument(
        "--recognizer",
        type=str,
        default="lbph",
        choices=["lbph", "mobilefacenet", "dlib"],
        help="Model pengenalan wajah: 'lbph' (default), 'mobilefacenet', atau 'dlib'"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help=f"Indeks perangkat kamera (default: {config.CAMERA_INDEX})"
    )
    parser.add_argument(
        "--user-name",
        type=str,
        default=config.DEFAULT_NAME,
        help="Nama identitas subjek untuk pendaftaran sampel wajah"
    )
    parser.add_argument(
        "--ear-thresh",
        type=float,
        default=config.EAR_THRESHOLD,
        help=f"Ambang batas EAR (default: {config.EAR_THRESHOLD})"
    )
    parser.add_argument(
        "--log-csv",
        action="store_true",
        default=config.CSV_LOG_ENABLED,
        help="Aktifkan pencatatan metrik performa ke file CSV"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=config.DISPLAY_SCALE,
        help="Faktor skala jendela display GUI (default: 2.0 untuk 640x480)"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Jalankan dalam mode headless tanpa jendela GUI (untuk server/benchmark murni)"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Import library CV & subsistem
    try:
        import cv2
        from video_stream import VideoStreamThread
        from landmarks import get_landmark_detector
        from recognition import get_face_recognizer
        from metrics import PerformanceMonitor, OSDRenderer
    except ImportError as err:
        print(f"[FATAL] Modul dependensi belum lengkap: {err}")
        print("Silakan jalankan: pip install -r requirements.txt")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(" SISTEM AUTENTIKASI WAJAH + LIVENESS DETECTION (RASPBERRY PI 3)")
    print("=" * 65)
    print(f" [*] Landmark Detector  : {args.detector.upper()}")
    print(f" [*] Face Recognizer    : {args.recognizer.upper()}")
    print(f" [*] Resolusi Input     : {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    print(f" [*] Ambang Batas EAR   : {args.ear_thresh}")
    print(f" [*] Logging CSV        : {'AKTIF' if args.log_csv else 'NON-AKTIF'}")
    print("=" * 65 + "\n")

    # 1. INISIALISASI STREAM KAMERA MULTITHREADING
    try:
        vs = VideoStreamThread(
            src=args.camera,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            fps=config.TARGET_FPS
        ).start()
    except Exception as e:
        print(f"[FATAL] Gagal memulai thread kamera: {e}")
        sys.exit(1)

    # 2. INISIALISASI DETEKTOR LANDMARK
    try:
        landmark_detector = get_landmark_detector(
            backend=args.detector,
            predictor_path=config.DLIB_PREDICTOR_PATH,
            task_model_path=config.FACE_LANDMARKER_MODEL_PATH
        )
    except Exception as e:
        print(f"[FATAL] Gagal menginisialisasi Landmark Detector: {e}")
        vs.stop()
        sys.exit(1)

    # 3. INISIALISASI FACE RECOGNIZER
    try:
        if args.recognizer == "lbph":
            recognizer = get_face_recognizer(
                backend="lbph",
                model_path=config.LBPH_MODEL_PATH,
                labels_path=config.LBPH_LABELS_PATH,
                distance_threshold=config.LBPH_DISTANCE_THRESHOLD
            )
        elif args.recognizer == "mobilefacenet":
            recognizer = get_face_recognizer(
                backend="mobilefacenet",
                model_path=config.MOBILEFACENET_MODEL_PATH,
                db_path=config.MOBILEFACENET_DB_PATH,
                threshold=config.MOBILEFACENET_THRESHOLD
            )
        else: # dlib
            recognizer = get_face_recognizer(
                backend="dlib",
                encodings_path=config.DLIB_ENCODINGS_PATH,
                tolerance=config.DLIB_TOLERANCE
            )
    except Exception as e:
        print(f"[FATAL] Gagal menginisialisasi Face Recognizer: {e}")
        landmark_detector.close()
        vs.stop()
        sys.exit(1)

    # 4. INISIALISASI METRICS MONITOR & OSD RENDERER
    exp_tag = f"{args.detector}_{args.recognizer}"
    monitor = PerformanceMonitor(
        log_to_csv=args.log_csv,
        log_dir=config.LOGS_DIR,
        exp_name=exp_tag
    )
    renderer = OSDRenderer(display_scale=args.scale)

    # VARIABEL STATE MACHINE LIVENESS & RECOGNITION
    current_state = STATE_NO_FACE
    consecutive_closed = 0
    blink_count = 0
    last_face_seen_time = 0.0
    liveness_start_time = 0.0
    decision_hold_start = 0.0

    recognized_name = "Unknown"
    recognized_dist = 1.0
    recog_time_ms = 0.0

    popup_message = ""
    popup_until = 0.0

    print("\n[PETUNJUK PENGGUNAAN]:")
    print("  [e] : Daftarkan wajah Anda ke database (ambil 3-5 sampel)")
    print("  [r] : Reset proses verifikasi liveness & autentikasi")
    print("  [s] : Aktifkan / Non-aktifkan pencatatan metrik CSV")
    print("  [q] : Keluar dari program\n")

    window_name = "Autentikasi Wajah + Liveness Detection (Skripsi)"
    if not args.no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Catat waktu awal per frame
            frame_t_start = time.perf_counter()

            # Tick monitor untuk update FPS dan beban CPU/RAM
            monitor.tick()

            # Ambil frame kamera terbaru (non-blocking)
            grabbed, frame = vs.read()
            if not grabbed or frame is None:
                time.sleep(0.005)
                continue

            now = time.time()

            # ------------------------------------------------------------------
            # TAHAP A: DETEKSI LANDMARK & PERHITUNGAN EAR (LIVENESS PRE-CHECK)
            # ------------------------------------------------------------------
            landmark_res = landmark_detector.detect(frame)
            landmark_time_ms = landmark_res.inference_time_ms

            # ------------------------------------------------------------------
            # TAHAP B: STATE MACHINE LIVENESS-FIRST
            # ------------------------------------------------------------------
            if not landmark_res.face_detected:
                # Wajah tidak terlihat
                if (now - last_face_seen_time) > config.RESET_FACE_LOST_SEC:
                    current_state = STATE_NO_FACE
                    consecutive_closed = 0
                    blink_count = 0
                    recognized_name = "Unknown"
                    recognized_dist = 1.0

                status_title = "ARAHKAN WAJAH KE DEPAN KAMERA"
                status_subtext = "Menunggu subjek wajah terdeteksi..."
                status_color = (150, 150, 150) # Abu-abu
            else:
                last_face_seen_time = now

                # Jika saat ini berada pada state keputusan (Granted / Denied)
                if current_state in (STATE_ACCESS_GRANTED, STATE_ACCESS_DENIED):
                    if (now - decision_hold_start) >= config.HOLD_DECISION_SEC:
                        # Waktu tahan habis, reset kembali untuk verifikasi berikutnya
                        current_state = STATE_LIVENESS_CHECKING
                        blink_count = 0
                        consecutive_closed = 0
                        liveness_start_time = now
                    else:
                        if current_state == STATE_ACCESS_GRANTED:
                            status_title = f"AKSES DITERIMA: {recognized_name.upper()}"
                            status_subtext = f"Liveness Terverifikasi (Dist: {recognized_dist:.2f})"
                            status_color = (0, 220, 0) # Hijau
                        else:
                            status_title = "AKSES DITOLAK: WAJAH TIDAK DIKENAL"
                            status_subtext = f"Identitas tidak cocok (Dist: {recognized_dist:.2f})"
                            status_color = (0, 0, 240) # Merah

                # Jika belum lolos liveness, lakukan evaluasi kedipan (EAR)
                elif current_state in (STATE_NO_FACE, STATE_LIVENESS_CHECKING):
                    if current_state == STATE_NO_FACE:
                        current_state = STATE_LIVENESS_CHECKING
                        liveness_start_time = now
                        blink_count = 0
                        consecutive_closed = 0

                    ear = landmark_res.ear_avg

                    # LOGIKA SIKLUS KEDIPAN MATA (Terbuka -> Terpejam -> Terbuka)
                    if ear < args.ear_thresh:
                        consecutive_closed += 1
                    else:
                        # Mata kembali terbuka, periksa durasi frame terpejam sebelumnya
                        if config.MIN_CLOSED_FRAMES <= consecutive_closed <= config.MAX_CLOSED_FRAMES:
                            blink_count += 1
                            popup_message = f"Kedipan Sah Terdeteksi! ({blink_count}/{config.REQUIRED_BLINKS})"
                            popup_until = now + 1.2
                            print(f"[INFO] Kedipan valid terdeteksi (Frame terpejam: {consecutive_closed}).")
                        consecutive_closed = 0

                    # Cek batas waktu liveness (timeout)
                    if (now - liveness_start_time) > config.LIVENESS_TIMEOUT_SEC:
                        blink_count = 0
                        liveness_start_time = now
                        popup_message = "Waktu kedip habis. Ulangi kedipan!"
                        popup_until = now + 1.5

                    # Periksa apakah syarat kedipan sudah terpenuhi
                    if blink_count >= config.REQUIRED_BLINKS:
                        # STATUS LIVENESS RESMI "PASSED"
                        current_state = STATE_LIVENESS_PASSED
                    else:
                        status_title = "UJI KEASLIAN (LIVENESS): SILAKAN BERKEDIP"
                        status_subtext = f"Kedipkan mata 1x untuk membuktikan orang asli (EAR: {ear:.2f})"
                        status_color = (0, 215, 255) # Kuning / Emas

                # --------------------------------------------------------------
                # TAHAP C: PENGENALAN WAJAH (HANYA DIEKSEKUSI JIKA LOKAL PASSED!)
                # --------------------------------------------------------------
                if current_state == STATE_LIVENESS_PASSED:
                    status_title = "LIVENESS VALID: MENCOCOKKAN IDENTITAS..."
                    status_subtext = "Memverifikasi fitur biometrik wajah..."
                    status_color = (255, 200, 0) # Cyan

                    # Eksekusi Face Recognition
                    if recognizer.get_sample_count() == 0:
                        current_state = STATE_ACCESS_DENIED
                        recognized_name = "Database Kosong"
                        recognized_dist = 1.0
                        decision_hold_start = now
                        popup_message = "Database wajah masih kosong! Tekan 'e' untuk daftar."
                        popup_until = now + 2.5
                    elif landmark_res.face_crop is not None:
                        recog_res = recognizer.recognize(landmark_res.face_crop)
                        recog_time_ms = recog_res.inference_time_ms
                        recognized_dist = recog_res.distance
                        decision_hold_start = now

                        if recog_res.is_match:
                            current_state = STATE_ACCESS_GRANTED
                            recognized_name = recog_res.name
                            print(f"[AKSES DITERIMA] Subjek: {recognized_name} | Dist: {recognized_dist:.2f} | Latensi: {recog_time_ms:.1f}ms")
                        else:
                            current_state = STATE_ACCESS_DENIED
                            recognized_name = "Unknown"
                            print(f"[AKSES DITOLAK] Wajah tidak dikenal | Dist: {recognized_dist:.2f} | Latensi: {recog_time_ms:.1f}ms")

            # Hitung total waktu proses frame ini
            total_pipeline_ms = (time.perf_counter() - frame_t_start) * 1000.0

            # ------------------------------------------------------------------
            # TAHAP D: PENCATATAN METRIK KE FILE CSV
            # ------------------------------------------------------------------
            monitor.log_frame(
                landmark_ms=landmark_time_ms,
                recog_ms=recog_time_ms,
                total_ms=total_pipeline_ms,
                ear_val=landmark_res.ear_avg,
                consecutive_closed=consecutive_closed,
                blink_count=blink_count,
                liveness_status=current_state,
                recog_name=recognized_name,
                distance_score=recognized_dist
            )

            # ------------------------------------------------------------------
            # TAHAP E: PENGGAMBARAN OSD & TAMPILAN GRAFIS
            # ------------------------------------------------------------------
            if not args.no_gui:
                display_img = renderer.draw_hud(
                    img=frame,
                    monitor=monitor,
                    landmark_res=landmark_res,
                    liveness_state_name=current_state,
                    status_text=status_title,
                    status_subtext=status_subtext,
                    status_color=status_color,
                    ear_threshold=args.ear_thresh,
                    required_blinks=config.REQUIRED_BLINKS,
                    current_blinks=blink_count,
                    sample_count=recognizer.get_sample_count(),
                    active_recognizer_name=args.recognizer
                )

                # Notifikasi Pop-up banner di tengah layar
                if now < popup_until and popup_message:
                    disp_h, disp_w = display_img.shape[:2]
                    pop_w = int(disp_w * 0.85)
                    pop_x = (disp_w - pop_w) // 2
                    pop_y = 50
                    cv2.rectangle(display_img, (pop_x, pop_y), (pop_x + pop_w, pop_y + 26), (0, 140, 255), -1)
                    cv2.rectangle(display_img, (pop_x, pop_y), (pop_x + pop_w, pop_y + 26), (255, 255, 255), 1)
                    cv2.putText(display_img, popup_message, (pop_x + 10, pop_y + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow(window_name, display_img)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
                # Beri sedikit jeda di mode no-gui agar tidak membakar core
                time.sleep(0.01)

            # ------------------------------------------------------------------
            # TAHAP F: PENANGANAN TOMBOL KONTROL KEYBOARD
            # ------------------------------------------------------------------
            if key == ord('q') or key == 27:
                print("\n[INFO] Pengguna menghentikan program.")
                break

            elif key == ord('e'):
                # DAFTARKAN SAMPEL WAJAH BARU
                if landmark_res.face_detected and landmark_res.face_crop is not None:
                    sukses = recognizer.enroll(landmark_res.face_crop, args.user_name)
                    if sukses:
                        count = recognizer.get_sample_count()
                        popup_message = f"Sampel {count} tersimpan untuk {args.user_name}!"
                        popup_until = now + 2.0
                        print(f"[SUKSES] Sampel wajah ke-{count} berhasil didaftarkan untuk '{args.user_name}'.")
                    else:
                        popup_message = "Gagal memproses fitur wajah. Coba lagi!"
                        popup_until = now + 2.0
                else:
                    popup_message = "Tidak ada wajah untuk didaftarkan!"
                    popup_until = now + 2.0

            elif key == ord('r'):
                # RESET STATE VERIFIKASI
                current_state = STATE_LIVENESS_CHECKING
                blink_count = 0
                consecutive_closed = 0
                liveness_start_time = now
                recognized_name = "Unknown"
                recognized_dist = 1.0
                popup_message = "Status verifikasi di-reset."
                popup_until = now + 1.5
                print("[INFO] Status verifikasi di-reset ke awal.")

            elif key == ord('s'):
                # TOGGLE PENCATATAN CSV METRIK
                is_active = monitor.toggle_csv_logging()
                if is_active:
                    popup_message = "Perekaman metrik CSV: AKTIF"
                else:
                    popup_message = "Perekaman metrik CSV: BERHENTI"
                popup_until = now + 2.0

    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan melalui terminal (Ctrl+C).")

    finally:
        # PEMBERSIHAN SUMBER DAYA SISTEM (CLEANUP)
        print("[INFO] Membersihkan dan menutup sumber daya...")
        monitor.close()
        landmark_detector.close()
        vs.stop()
        if not args.no_gui:
            cv2.destroyAllWindows()
        print("[INFO] Program selesai dengan aman.")


if __name__ == "__main__":
    main()
