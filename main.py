#!/usr/bin/env python3
"""
Face Recognition + Liveness Detection (Eye Aspect Ratio / kedipan mata)
Dioptimalkan untuk Raspberry Pi 3.

Cara pakai:
  1. Daftarkan wajah   : python3 face_liveness.py enroll --name Budi
  2. Jalankan verifikasi: python3 face_liveness.py run
     (tambahkan --no-display jika Pi dijalankan tanpa monitor)

Logika:
  - Wajah dikenali lewat face_recognition (dlib, embedding 128-d).
  - Liveness: EAR dihitung dari 6 titik landmark tiap mata.
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Saat mata terbuka EAR ~0.25-0.35, saat berkedip turun < ~0.21.
  - Status "ASLI" hanya jika wajah dikenali DAN ada kedipan dalam
    beberapa detik terakhir.
"""

import argparse
import os
import pickle
import time

import cv2
import numpy as np
import face_recognition

# ----------------------------- KONFIGURASI -----------------------------
ENCODINGS_FILE = "encodings.pickle"

CAM_INDEX = 0            # index webcam USB (0 = default)
FRAME_W, FRAME_H = 320, 240   # resolusi kecil agar ringan di Pi 3
DETECT_SCALE = 1.0       # turunkan (mis. 0.75) jika masih lambat

EAR_THRESHOLD = 0.21     # EAR di bawah ini dianggap mata tertutup
EAR_CONSEC_FRAMES = 2    # minimal frame berurutan mata tertutup = 1 kedipan
BLINK_VALID_SEC = 10     # kedipan dianggap valid selama N detik
MIN_BLINKS = 1           # jumlah kedipan minimal agar dianggap asli

MATCH_TOLERANCE = 0.50   # makin kecil makin ketat (default library 0.6)
RECOGNIZE_EVERY = 1.0    # detik, interval encoding wajah (berat di Pi)
LOST_FACE_RESET = 1.5    # detik tanpa wajah -> reset status
# -----------------------------------------------------------------------


def eye_aspect_ratio(eye_points):
    """Hitung EAR dari 6 titik landmark mata."""
    p = np.array(eye_points, dtype=np.float64)
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak bisa dibuka. Cek CAM_INDEX / koneksi.")
    return cap


def largest_face(locations):
    """Ambil wajah terbesar (top, right, bottom, left)."""
    return max(locations, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))


def load_db():
    if not os.path.exists(ENCODINGS_FILE):
        return {}
    with open(ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)


def save_db(db):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(db, f)


# ------------------------------- ENROLL --------------------------------
def enroll(name, samples=5):
    cap = open_camera()
    print(f"Pendaftaran '{name}'. Hadapkan wajah ke kamera.")
    for i in range(3, 0, -1):
        print(f"Mulai dalam {i}...")
        t0 = time.time()
        while time.time() - t0 < 1:
            cap.read()

    encodings = []
    last = 0
    while len(encodings) < samples:
        ok, frame = cap.read()
        if not ok:
            continue
        if time.time() - last < 0.7:     # beri jeda antar sampel
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        if len(locs) != 1:
            print("Harus tepat 1 wajah terlihat...")
            last = time.time()
            continue
        enc = face_recognition.face_encodings(rgb, locs)[0]
        encodings.append(enc)
        last = time.time()
        print(f"Sampel {len(encodings)}/{samples} tersimpan")

    cap.release()
    db = load_db()
    db.setdefault(name, []).extend(encodings)
    save_db(db)
    print(f"Selesai. Data disimpan di {ENCODINGS_FILE}")


# -------------------------------- RUN ----------------------------------
def identify(rgb, loc, db):
    """Cocokkan wajah dengan database. Return nama atau 'Unknown'."""
    enc = face_recognition.face_encodings(rgb, [loc])
    if not enc or not db:
        return "Unknown"
    best_name, best_dist = "Unknown", 1.0
    for name, known in db.items():
        d = face_recognition.face_distance(known, enc[0]).min()
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name if best_dist <= MATCH_TOLERANCE else "Unknown"


def run(display=True):
    db = load_db()
    if not db:
        print("Database kosong. Jalankan 'enroll' dulu.")
        return

    cap = open_camera()
    name = "Unknown"
    last_recognize = 0
    last_seen = time.time()

    closed_frames = 0
    blink_times = []          # timestamp tiap kedipan
    ear_value = 0.0
    prev_status = None

    print("Berjalan. Tekan 'q' untuk keluar (atau Ctrl+C).")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.time()

            small = frame if DETECT_SCALE == 1.0 else cv2.resize(
                frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locs = face_recognition.face_locations(rgb, model="hog")
            loc = None

            if locs:
                last_seen = now
                loc = largest_face(locs)

                # --- Landmark & EAR (dihitung tiap frame, ringan) ---
                lm = face_recognition.face_landmarks(rgb, [loc])
                if lm:
                    l_ear = eye_aspect_ratio(lm[0]["left_eye"])
                    r_ear = eye_aspect_ratio(lm[0]["right_eye"])
                    ear_value = (l_ear + r_ear) / 2.0

                    if ear_value < EAR_THRESHOLD:
                        closed_frames += 1
                    else:
                        if closed_frames >= EAR_CONSEC_FRAMES:
                            blink_times.append(now)
                        closed_frames = 0

                # --- Pengenalan wajah (berat, dijalankan berkala) ---
                if now - last_recognize >= RECOGNIZE_EVERY:
                    name = identify(rgb, loc, db)
                    last_recognize = now
            elif now - last_seen > LOST_FACE_RESET:
                name = "Unknown"
                blink_times.clear()
                closed_frames = 0
                ear_value = 0.0

            # Buang kedipan yang sudah kedaluwarsa
            blink_times = [t for t in blink_times if now - t <= BLINK_VALID_SEC]

            # --- Keputusan akhir ---
            if loc is None:
                status = "TIDAK ADA WAJAH"
            elif name == "Unknown":
                status = "WAJAH TIDAK DIKENAL"
            elif len(blink_times) >= MIN_BLINKS:
                status = f"ASLI: {name}"
            else:
                status = f"{name} - SILAKAN BERKEDIP"

            if status != prev_status:
                print(f"[{time.strftime('%H:%M:%S')}] {status} "
                      f"(EAR={ear_value:.2f}, kedip={len(blink_times)})")
                prev_status = status

            if display:
                if loc is not None:
                    t, r, b, l = [int(v / DETECT_SCALE) for v in loc]
                    color = (0, 200, 0) if status.startswith("ASLI") else (0, 165, 255)
                    cv2.rectangle(frame, (l, t), (r, b), color, 2)
                cv2.putText(frame, status, (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"EAR: {ear_value:.2f}  Kedip: {len(blink_times)}",
                            (5, FRAME_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 0), 1)
                cv2.imshow("Face Liveness", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


# -------------------------------- MAIN ---------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Face recognition + liveness (EAR)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_enroll = sub.add_parser("enroll", help="daftarkan wajah baru")
    p_enroll.add_argument("--name", required=True)
    p_enroll.add_argument("--samples", type=int, default=5)

    p_run = sub.add_parser("run", help="jalankan verifikasi")
    p_run.add_argument("--no-display", action="store_true",
                       help="tanpa jendela video (headless)")

    args = ap.parse_args()
    if args.cmd == "enroll":
        enroll(args.name, args.samples)
    else:
        run(display=not args.no_display)