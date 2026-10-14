# Smart Door Lock - Liveness Detection System

**Judul Skripsi:** "Deteksi Liveness Berbasis Kedipan Mata untuk Autentikasi Wajah pada Smart Door Lock Berbasis Raspberry Pi"

Prototype sistem yang menggunakan deteksi kedipan mata (blink detection) sebagai anti-spoofing untuk autentikasi wajah pada smart door lock.

---

## 📋 Daftar Isi

1. [Fitur Utama](#fitur-utama)
2. [Prerequisite](#prerequisite)
3. [Instalasi](#instalasi)
4. [Struktur Project](#struktur-project)
5. [Cara Menggunakan](#cara-menggunakan)
6. [Penjelasan Algoritma](#penjelasan-algoritma)
7. [Troubleshooting](#troubleshooting)
8. [Modifikasi untuk Penelitian](#modifikasi-untuk-penelitian)

---

## 🎯 Fitur Utama

### 1. **Face Detection (Deteksi Wajah)**
- Menggunakan **MediaPipe Face Mesh** untuk deteksi dan pelacakan wajah real-time
- Ekstraksi 468 facial landmarks dengan akurasi tinggi
- Lightweight dan efisien untuk embedded systems (Raspberry Pi)

### 2. **Liveness Detection (Deteksi Kedipan Mata)**
- **Eye Aspect Ratio (EAR)** untuk mendeteksi pembukaan/penutupan mata
- Anti-spoofing: Mencegah akses menggunakan foto atau video
- Konfigurasi threshold dan sensitivity yang mudah di-tune

### 3. **Face Recognition (Pengenalan Wajah)**
- Menggunakan library `face_recognition` (berbasis dlib)
- Encoding wajah dengan akurasi tinggi (128-dimensional vector)
- Database lokal dengan multiple photos per person untuk meningkatkan akurasi
- Matching dengan tolerance threshold yang dapat dikonfigurasi

### 4. **Real-time Performance Monitoring**
- FPS counter untuk monitoring performa
- Display EAR values untuk debug
- Status liveness dan recognition result

---

## 📦 Prerequisite

### Hardware
- PC/Laptop dengan webcam
- Python 3.7+

### Software
- Python 3.7+
- pip (Python package manager)

---

## 🚀 Instalasi

### Step 1: Clone atau Download Project

```bash
cd ~/Projects  # atau lokasi project Anda
# Download atau git clone project ini
```

### Step 2: Buat Virtual Environment (Recommended)

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Catatan untuk Linux:**
Jika ada error saat install `face-recognition`, mungkin perlu install dependencies sistem:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install cmake python3-dev

# Untuk older versions of dlib
pip install cmake
```

### Step 4: Siapkan Database Wajah

```bash
# Folder akan dibuat otomatis jika belum ada, tapi bisa dibuat manual
mkdir face_database

# Masukkan foto wajah ke folder
# Contoh:
# - face_database/john_doe.jpg
# - face_database/john_doe_2.jpg (optional, untuk lebih banyak encoding)
# - face_database/jane_smith.jpg
```

**Tips untuk foto terbaik:**
- Minimal 100x100 pixels
- Wajah terlihat jelas dan frontal
- Pencahayaan cukup baik
- Format: JPG, PNG, atau BMP
- Bisa ada multiple photos untuk 1 orang (misal: john_doe_1.jpg, john_doe_2.jpg)

### Step 5: Jalankan Program

```bash
python main.py
```

---

## 📁 Struktur Project

```
smart_door_lock/
├── main.py                      # Script utama
├── config.py                    # Konfigurasi parameter
├── requirements.txt             # Dependencies
├── SETUP_INSTRUCTIONS.md        # Instruksi setup
├── README.md                    # File ini
│
├── face_database/               # Database wajah (di-create otomatis)
│   ├── person_1.jpg
│   └── person_2.jpg
│
├── modules/                     # Custom modules (modular architecture)
│   ├── __init__.py
│   ├── face_detector.py         # Face detection + facial landmarks
│   ├── liveness_detector.py     # Blink detection + liveness
│   ├── face_recognizer.py       # Face recognition + matching
│   └── utils.py                 # Utility functions (FPS, display, etc)
│
└── logs/                        # Log files (optional)
```

---

## 💻 Cara Menggunakan

### Running the Program

```bash
python main.py
```

Output akan menampilkan:
1. Informasi database yang ter-load
2. Real-time video dengan annotations
3. Status system (FPS, EAR values, liveness, recognition result)

### Keyboard Controls

| Key | Fungsi |
|-----|--------|
| `q` | Quit program |
| `s` | Save current frame sebagai debug image |
| `r` | Reset liveness detection |

### Display Information

**Di layar akan menampilkan:**

1. **EAR Values** (top-left)
   - EAR Left dan EAR Right (real-time)
   - Threshold EAR yang dikonfigurasi

2. **Liveness Status** (left side)
   - "Liveness: Silakan Berkedip" (belum ada blink)
   - "Liveness: ASLI (Real)" (blink terdeteksi)

3. **Recognition Result** (bottom)
   - "PINTU TERBUKA - Selamat Datang [Nama]!" (Recognized)
   - "AKSES DITOLAK - Wajah Tidak Dikenali" (Unknown)
   - Confidence score

4. **Status Panel** (top-right)
   - Jumlah face yang terdeteksi
   - Average EAR
   - Threshold value

5. **FPS Counter** (top-right corner)
   - Frame per second untuk monitoring performa

---

## 📊 Penjelasan Algoritma

### 1. Eye Aspect Ratio (EAR) untuk Blink Detection

**Formula:**
```
EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
```

Dimana:
- P1: Inner corner (sudut dalam mata)
- P2, P3: Top eyelid (kelopak atas)
- P4: Outer corner (sudut luar mata)
- P5, P6: Bottom eyelid (kelopak bawah)

**Interpretasi:**
- Mata terbuka: EAR ≈ 0.40-0.50
- Mata tertutup: EAR ≈ 0.10-0.15
- **Threshold** (default): 0.25
  - Jika EAR < 0.25 → Mata dianggap tertutup
  - Jika EAR >= 0.25 → Mata dianggap terbuka

**Deteksi Blink:**
1. Monitor EAR terus-menerus
2. Jika EAR turun < threshold selama N frame berturut-turut (default: 2 frame) → Terdeteksi mulai blink
3. Jika EAR naik >= threshold → Blink selesai
4. Counter blink increment
5. Jika blink_count >= BLINKS_REQUIRED (default: 1) → Liveness = True

### 2. Face Recognition Process

**Step-by-step:**
1. Load semua gambar dari `face_database/`
2. Detect wajah dalam setiap gambar
3. Generate face encoding (128-dimensional vector) untuk setiap wajah
4. Simpan encodings dan nama person

**Recognition:**
1. Capture wajah dari webcam
2. Generate encoding
3. Compare dengan semua encoding di database
4. Hitung Euclidean distance
5. Jika distance < tolerance threshold → Match
6. Confidence = 1 - distance

---

## ⚙️ Konfigurasi Parameter

Edit file `config.py` untuk mengatur parameter:

```python
# Liveness Detection
EAR_THRESHOLD = 0.25                    # Sensitivity blink detection
CONSECUTIVE_FRAMES_BLINK = 2            # Frame untuk confirm blink
BLINKS_REQUIRED = 1                     # Jumlah blink yang diperlukan

# Face Recognition
FACE_RECOGNITION_TOLERANCE = 0.6        # Strict/permissive untuk matching
```

### Rekomendasi Tuning

**Untuk environment dengan pencahayaan baik:**
- `EAR_THRESHOLD = 0.22` (lebih sensitif)
- `CONSECUTIVE_FRAMES_BLINK = 1`

**Untuk environment dengan pencahayaan buruk:**
- `EAR_THRESHOLD = 0.28` (lebih permisif)
- `CONSECUTIVE_FRAMES_BLINK = 3` (lebih ketat)

**Untuk face recognition lebih strict:**
- `FACE_RECOGNITION_TOLERANCE = 0.4` (lebih ketat)

**Untuk face recognition lebih permisif:**
- `FACE_RECOGNITION_TOLERANCE = 0.7` (lebih permisif)

---

## 🔧 Troubleshooting

### Issue 1: "No module named 'cv2'"

**Solution:**
```bash
pip install --upgrade opencv-python
```

### Issue 2: "No module named 'mediapipe'"

**Solution:**
```bash
# Untuk PC biasa
pip install mediapipe

# Untuk Raspberry Pi (versi ringan)
pip install mediapipe-rpi
```

### Issue 3: Webcam tidak terdeteksi

**Solution:**
```python
# Di config.py, coba ubah CAMERA_INDEX
CAMERA_INDEX = 0  # Coba 0, 1, 2, dst
```

### Issue 4: Face detection terlalu lambat

**Solution:**
- Turunkan resolution di config.py:
  ```python
  CAMERA_WIDTH = 320   # Dari 640
  CAMERA_HEIGHT = 240  # Dari 480
  ```

### Issue 5: Blink detection tidak sensitif

**Solution:**
- Turunkan EAR_THRESHOLD:
  ```python
  EAR_THRESHOLD = 0.20  # Dari 0.25
  ```
- Atau kurangi CONSECUTIVE_FRAMES_BLINK:
  ```python
  CONSECUTIVE_FRAMES_BLINK = 1  # Dari 2
  ```

---

## 🔬 Modifikasi untuk Penelitian

Script ini dirancang untuk mudah dimodifikasi. Berikut cara untuk swap berbagai komponen:

### 1. Menggunakan Model Lain untuk Face Detection

**Saat ini:** MediaPipe Face Mesh

**Alternatif:** OpenCV DNN, YOLO, RetinaFace, etc

Modifikasi di `modules/face_detector.py`:
```python
# Replace MediaPipe dengan model lain di __init__ method
```

### 2. Menggunakan Algoritma Liveness Lain

**Saat ini:** Eye Aspect Ratio (EAR)

**Alternatif:** Texture Analysis, Frequency Analysis, Deep Learning models

Modifikasi di `modules/liveness_detector.py`:
- Buat method `calculate_texture()` atau `calculate_frequency()`
- Update `detect_blink()` method

### 3. Menggunakan Model Face Recognition Lain

**Saat ini:** face_recognition (dlib)

**Alternatif:** VGGFace2, ArcFace, FaceNet, DeepFace

Modifikasi di `modules/face_recognizer.py`:
- Replace encoding generation
- Update distance calculation

### 4. Menambah Comparison Logic

Untuk meneliti performa berbagai model:

```python
# main.py - SmartDoorLockSystem.process_frame()

# Model 1
result_model1, conf1 = recognizer1.recognize_face(encoding)

# Model 2
result_model2, conf2 = recognizer2.recognize_face(encoding)

# Compare dan log hasil
results['model1'] = (result_model1, conf1)
results['model2'] = (result_model2, conf2)
```

---

## 📈 Performance Tips untuk Raspberry Pi

1. **Reduce frame resolution:**
   ```python
   CAMERA_WIDTH = 320
   CAMERA_HEIGHT = 240
   ```

2. **Use MediaPipe Lite:**
   ```bash
   pip install mediapipe-rpi
   ```

3. **Increase detection interval:**
   ```python
   # Skip beberapa frame untuk detection
   ```

4. **Optimize model:**
   - Use quantized models
   - Reduce max_num_faces

---

## 📝 Citation untuk Skripsi

Jika menggunakan script ini untuk skripsi, cite:

**Libraries:**
- MediaPipe: https://github.com/google/mediapipe
- face_recognition: https://github.com/ageitgey/face_recognition
- OpenCV: https://opencv.org

**Papers:**
- Tereza Soukupová and Jan Čech. "Real-Time Eye Blink Detection using Facial Landmarks." 2016
- Aleksandr Mordvintsev, et al. "OpenCV Face Recognition." 2015

---

## 📄 License

Script ini dibuat untuk keperluan skripsi/penelitian. Silakan modifikasi sesuai kebutuhan.

---

## 📧 Notes untuk Development

- [x] Face detection dengan MediaPipe
- [x] Liveness detection dengan EAR
- [x] Face recognition dengan face_recognition
- [x] FPS monitoring
- [x] Modular architecture
- [x] Configuration file
- [ ] Database persistence (bisa ditambah)
- [ ] Logging system (bisa ditambah)
- [ ] Web interface (bisa ditambah untuk future)
- [ ] Raspberry Pi optimization (bisa ditambah)

---

**Happy researching! 🚀**
