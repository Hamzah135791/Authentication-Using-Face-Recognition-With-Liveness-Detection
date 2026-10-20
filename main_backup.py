import os
import pickle
import time

import cv2
import numpy as np
import face_recognition

# ---------------- PENGATURAN ----------------
NAMA = "Hamzah"            # nama yang disimpan saat tekan tombol 'e'
FILE_DB = "en   codings.pickle"
CAM_INDEX = 1              # coba 1 atau 2 jika kamera salah/hitam
EAR_THRESHOLD = 0.21       # EAR di bawah ini = mata tertutup
EAR_FRAMES = 2             # frame berurutan mata tertutup = 1 kedipan
BLINK_VALID = 10           # detik kedipan dianggap valid
TOLERANCE = 0.50           # makin kecil makin ketat

DETECT_SCALE = 0.5         # deteksi di gambar 50% (lebih cepat)
UPSAMPLE = 1               # 1 = cari wajah lebih teliti (0 = cepat tapi sering miss)
HOLD_SEC = 0.6             # tahan kotak wajah sekian detik jika deteksi hilang sesaat
RESET_SEC = 1.5            # reset status jika wajah benar-benar hilang selama ini
# --------------------------------------------

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def perbaiki_cahaya(bgr):
    """Ratakan kontras agar deteksi lebih stabil di cahaya kurang bagus."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def hitung_ear(eye):
    p = np.array(eye, dtype=float)
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return (a + b) / (2.0 * c) if c else 0.0


# Buka kamera
if os.name == "nt":
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("Kamera tidak bisa dibuka. Ganti CAM_INDEX.")

# Muat database wajah (jika ada)
known = []
if os.path.exists(FILE_DB):
    with open(FILE_DB, "rb") as f:
        known = pickle.load(f)

nama = "Unknown"
last_recog = 0
closed = 0
blinks = []
ear = 0.0
last_loc = None
last_seen = 0
pesan, pesan_until = "", 0
fps, t_prev, det_ms = 0.0, time.time(), 0.0

print("Tombol: [e] daftarkan wajah | [q] keluar")

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    now = time.time()

    rgb = cv2.cvtColor(perbaiki_cahaya(frame), cv2.COLOR_BGR2RGB)

    # --- Deteksi di gambar yang diperkecil, lalu koordinat dikembalikan ---
    small = cv2.resize(rgb, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
    t0 = time.time()
    locs_s = face_recognition.face_locations(
        small, number_of_times_to_upsample=UPSAMPLE, model="hog")
    det_ms = (time.time() - t0) * 1000
    locs = [tuple(int(v / DETECT_SCALE) for v in l) for l in locs_s]

    fresh = bool(locs)   # True = wajah terdeteksi di frame ini
    if locs:
        loc = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
        last_loc, last_seen = loc, now
    elif last_loc and now - last_seen <= HOLD_SEC:
        loc = last_loc   # tahan sebentar supaya tidak berkedip-kedip
    else:
        loc = None

    if now - last_seen > RESET_SEC:
        nama, closed, ear, last_loc = "Unknown", 0, 0.0, None
        blinks.clear()

    if loc and fresh:
        # --- EAR tiap frame ---
        lm = face_recognition.face_landmarks(rgb, [loc])
        if lm:
            ear = (hitung_ear(lm[0]["left_eye"]) + hitung_ear(lm[0]["right_eye"])) / 2
            if ear < EAR_THRESHOLD:
                closed += 1
            else:
                if closed >= EAR_FRAMES:
                    blinks.append(now)
                closed = 0

        # --- Pengenalan wajah tiap 0.5 detik ---
        if known and now - last_recog > 0.5:
            enc = face_recognition.face_encodings(rgb, [loc])
            if enc:
                d = face_recognition.face_distance(known, enc[0]).min()
                nama = NAMA if d <= TOLERANCE else "Unknown"
            last_recog = now

    blinks = [t for t in blinks if now - t <= BLINK_VALID]

    # --- Status akhir ---
    if not loc:
        status, warna = "TIDAK ADA WAJAH", (0, 0, 255)
    elif not known:
        status, warna = "BELUM TERDAFTAR - TEKAN 'e'", (0, 165, 255)
    elif nama == "Unknown":
        status, warna = "WAJAH TIDAK DIKENAL", (0, 0, 255)
    elif blinks:
        status, warna = f"ASLI: {nama}", (0, 200, 0)
    else:
        status, warna = f"{nama} - SILAKAN BERKEDIP", (0, 165, 255)

    # --- Tampilan ---
    if loc:
        t, r, b, l = loc
        cv2.rectangle(frame, (l, t), (r, b), warna, 2)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna, 2)
    cv2.putText(frame, f"Wajah: {len(locs)} | EAR: {ear:.2f} | Kedip: {len(blinks)}",
                (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- FPS (dihaluskan) dan waktu deteksi ---
    t_now = time.time()
    dt = t_now - t_prev
    t_prev = t_now
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
    cv2.putText(frame, f"FPS: {fps:.1f}", (500, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Deteksi: {det_ms:.0f} ms", (460, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if now < pesan_until:
        cv2.putText(frame, pesan, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    cv2.imshow("Face Liveness (e=daftar, q=keluar)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    if key == ord("e"):
        if len(locs) == 1:
            enc = face_recognition.face_encodings(rgb, locs)[0]
            known.append(enc)
            with open(FILE_DB, "wb") as f:
                pickle.dump(known, f)
            pesan = f"Tersimpan! ({len(known)} sampel) - tekan 'e' lagi 3-5x"
        else:
            pesan = f"Gagal: terdeteksi {len(locs)} wajah (harus 1)"
        pesan_until = now + 2.5

cap.release()
cv2.destroyAllWindows()