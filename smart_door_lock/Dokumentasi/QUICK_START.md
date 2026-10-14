# 🚀 Quick Start Guide

## Daftar File yang Dibuat

```
smart_door_lock/
├── 📄 main.py                      ⭐ SCRIPT UTAMA - Jalankan ini!
├── 📄 config.py                    ⚙️  Konfigurasi semua parameter
├── 📄 requirements.txt              📦 Instalasi dependencies
├── 📄 create_face_database.py      🖼️  Utility untuk buat database wajah
├── 📄 test_modules.py              🧪 Test individual modules
│
├── 📁 modules/                     🧩 Modul-modul terpisah (modular!)
│   ├── __init__.py
│   ├── face_detector.py            👤 Deteksi wajah (MediaPipe)
│   ├── liveness_detector.py        👁️  Deteksi blink (EAR)
│   ├── face_recognizer.py          🆔 Pengenalan wajah
│   └── utils.py                    🛠️  Helper functions
│
├── 📁 face_database/               💾 Database wajah (auto-created)
│   └── [masukkan foto wajah di sini]
│
├── 📖 README.md                    📚 Dokumentasi lengkap
├── 📖 SETUP_INSTRUCTIONS.md        ⚙️  Instruksi setup detail
├── 📖 RESEARCH_GUIDE.md            🔬 Panduan untuk swap model algoritma
└── 📖 QUICK_START.md               ⚡ File ini
```

---

## ⚡ Instalasi Cepat (5 Menit)

### Step 1: Install Python & Git (Skip jika sudah ada)
```bash
# Windows: Download dari python.org
# Linux/Mac: sudo apt-get install python3 python3-pip
```

### Step 2: Setup Project
```bash
# Navigate ke folder project
cd path/to/smart_door_lock

# Buat virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Tunggu hingga selesai (~2-5 menit tergantung internet).

### Step 4: Persiapkan Database
```bash
# Folder face_database/ akan di-create otomatis
# Tapi bisa juga buat manual:
mkdir face_database

# Copy/masukkan foto wajah ke folder:
# - john_doe.jpg
# - jane_smith.jpg
# dll
```

### Step 5: Jalankan Program
```bash
python main.py
```

**That's it!** Program akan langsung jalan dengan webcam. 🎉

---

## 🎮 Cara Pakai

### Display akan menunjukkan:
- **Top-left**: EAR values (Eye Aspect Ratio)
- **Left side**: Status liveness ("Silakan Berkedip" atau "ASLI")
- **Bottom**: Hasil recognition ("PINTU TERBUKA" atau "AKSES DITOLAK")
- **Top-right**: FPS dan status panel

### Keyboard Controls:
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save debug frame |
| `r` | Reset liveness status |

### Workflow:
1. **Face muncul** → "Detect faces: 1"
2. **Kedip mata** → "Liveness: ASLI (Real)"
3. **Recognition** → "PINTU TERBUKA" atau "AKSES DITOLAK"

---

## 🛠️ Utility Scripts

### 1. Buat Database Wajah dari Webcam
```bash
python create_face_database.py
```
**Instruksi:**
- Input nama orang
- Tekan 's' untuk save foto (ambil berbagai angle)
- Tekan 'c' untuk complete
- Tekan 'q' untuk quit

### 2. Test Individual Modules
```bash
python test_modules.py
```
**Menu:**
- Test Face Detector saja
- Test Liveness Detector saja
- Test Face Recognizer saja

Berguna untuk debugging dan validation.

---

## 🔧 Konfigurasi Cepat

Edit `config.py` untuk tune parameter:

### Jika Blink Detection Tidak Sensitif:
```python
EAR_THRESHOLD = 0.22              # Turun dari 0.25
CONSECUTIVE_FRAMES_BLINK = 1      # Turun dari 2
```

### Jika Terlalu Sensitif (False Positives):
```python
EAR_THRESHOLD = 0.28              # Naik dari 0.25
CONSECUTIVE_FRAMES_BLINK = 3      # Naik dari 2
```

### Jika Face Recognition Terlalu Strict:
```python
FACE_RECOGNITION_TOLERANCE = 0.7  # Naik dari 0.6
```

### Jika Ingin Lebih Cepat di Raspberry Pi:
```python
CAMERA_WIDTH = 320                # Turun dari 640
CAMERA_HEIGHT = 240               # Turun dari 480
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Webcam tidak terdeteksi | Cek driver, atau ubah `CAMERA_INDEX` di config |
| "ModuleNotFoundError: cv2" | `pip install --upgrade opencv-python` |
| Blink tidak terdeteksi | Lihat konfigurasi di atas, atau cek pencahayaan |
| Facial recognition error | Pastikan ada foto di `face_database/` |
| Program jalan lambat | Turunkan resolution atau skip frames |

---

## 📊 Untuk Skripsi - Performance Testing

### Collect Metrics Saat Running:
Program sudah menampilkan:
- ✅ FPS real-time
- ✅ EAR values (untuk analisis)
- ✅ Detection status
- ✅ Recognition confidence

Save ke CSV untuk analysis:
```python
# Bisa dimodif main.py untuk log ke file
with open('metrics.csv', 'w') as f:
    f.write("frame_no,fps,ear_left,ear_right,liveness,person,confidence\n")
    # Log setiap frame
```

### Perbandingan Model:
Lihat `RESEARCH_GUIDE.md` untuk cara swap model dan membandingkan.

---

## 💡 Tips & Tricks

1. **Foto Database Terbaik**: 
   - Wajah jelas dan frontal
   - Pencahayaan normal (bukan terlalu terang/gelap)
   - Minimal 1 foto per person, lebih banyak lebih akurat
   - Multiple angles bagus untuk robustness

2. **Blink Detection**:
   - Natural blink, jangan close eyes terlalu lama
   - Jika ada glasses, mungkin perlu tune threshold
   - Cahaya sidelight lebih susah dari frontal

3. **Performance Optimization**:
   - Untuk laptop biasa: gpp biar high resolution
   - Untuk Raspberry Pi: prioritaskan FPS vs resolution

4. **Debug**:
   - Tekan 's' untuk save frame untuk analisis
   - Gunakan `test_modules.py` untuk test individual komponen
   - Check console output jika DEBUG_MODE = True

---

## 📚 File Dokumentasi

Untuk info lebih detail:

- **`README.md`** - Dokumentasi lengkap, algoritma, API
- **`RESEARCH_GUIDE.md`** - Cara swap model, membandingkan algoritma
- **`SETUP_INSTRUCTIONS.md`** - Instalasi detail, troubleshooting
- **`config.py`** - Penjelasan setiap parameter

---

## 🎓 Untuk Skripsi

### Struktur yang Bisa Digunakan:

**Bab: Implementation**
- Arsitektur sistem (diagram included)
- Face detection dengan MediaPipe
- Liveness detection dengan EAR
- Face recognition dengan face_recognition

**Bab: Evaluation**
- Testing methodology
- Performance metrics (FPS, accuracy, inference time)
- Comparison dengan alternative methods

**Bab: Results**
- Performance table
- Graphs & charts
- Discussion

---

## ✅ Checklist Sebelum Demo/Submission

- [ ] Main script jalan lancar
- [ ] Database wajah sudah ter-load
- [ ] Blink detection berfungsi dengan baik
- [ ] Face recognition akurat
- [ ] FPS cukup tinggi (>15 FPS untuk webcam)
- [ ] Code ter-comment dengan baik
- [ ] Config file ter-dokumentasi
- [ ] Test modules jalan semua
- [ ] Database utility berfungsi
- [ ] README lengkap

---

## 🚀 Next Steps

1. **Sekarang**: Run `python main.py` dan test dengan wajah Anda
2. **Setup Database**: Gunakan `create_face_database.py` untuk tambah person
3. **Test Modules**: Gunakan `test_modules.py` untuk validate tiap komponen
4. **Tune Parameter**: Adjust `config.py` sesuai environment
5. **Untuk Raspberry Pi**: Nanti lanjut dengan deployment guide

---

## 📞 Support

Jika ada error atau pertanyaan:
1. Cek README.md untuk dokumentasi lengkap
2. Cek config.py untuk penjelasan parameter
3. Cek RESEARCH_GUIDE.md untuk info technical
4. Run test_modules.py untuk debug specific component

---

**Happy coding dan semoga skripsi sukses! 🎓✨**

Kalau ada pertanyaan atau perlu modifikasi lebih lanjut, jangan ragu untuk ask! 
