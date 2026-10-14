# Panduan Setup - Liveness Detection System untuk Smart Door Lock

## 1. Struktur Folder Project

```
smart_door_lock/
├── main.py                          # Script utama
├── requirements.txt                 # Dependency list
├── config.py                        # Konfigurasi parameter
├── face_database/                   # Database wajah yang sah
│   ├── person_1.jpg                # Foto orang 1 (bisa multiple foto)
│   ├── person_1_2.jpg
│   └── person_2.jpg                # Foto orang 2
├── modules/
│   ├── __init__.py
│   ├── face_detector.py             # Module deteksi wajah
│   ├── liveness_detector.py         # Module deteksi liveness (blink)
│   ├── face_recognizer.py           # Module face recognition
│   └── utils.py                     # Utility functions
└── logs/                            # Folder untuk logging (opsional)
```

## 2. Installation Steps

### Step 1: Buat Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# atau
venv\Scripts\activate             # Windows
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Siapkan Face Database
- Buat folder `face_database/` di root project
- Masukkan foto-foto wajah dengan penamaan konsisten:
  - Format: `nama_orang.jpg` atau `nama_orang_1.jpg`, `nama_orang_2.jpg`
  - Minimal 1 foto per person (lebih banyak lebih akurat)
  - Ukuran foto: minimal 100x100 px
  - Format: JPG, PNG, atau BMP

### Step 4: Update config.py
- Sesuaikan threshold EAR, jumlah frame untuk blink, dll sesuai kebutuhan

### Step 5: Jalankan Script
```bash
python main.py
```

## 3. Kebutuhan Library

| Library | Versi | Fungsi |
|---------|-------|--------|
| opencv-python | >=4.5.0 | Capture webcam & display |
| mediapipe | >=0.8.0 | Facial landmarks detection |
| face-recognition | >=1.3.0 | Face encoding & recognition |
| numpy | >=1.19.0 | Numerical computation |
| pillow | >=8.0.0 | Image processing |

## 4. Kontrol Keyboard

- **q**: Quit program
- **s**: Save current face frame untuk debug
- **r**: Reset liveness status

## 5. Troubleshooting

### Error: "No module named 'cv2'"
- Pastikan virtual env aktif dan pip install berhasil
- Coba: `pip install --upgrade opencv-python`

### Error: "No module named 'mediapipe'"
- Untuk Raspberry Pi: `pip install mediapipe-rpi`
- Untuk PC biasa: `pip install mediapipe`

### Webcam tidak terdeteksi
- Cek permission aksses kamera
- Coba ubah camera_index di config.py (0, 1, 2, dst)

### Face detection lambat
- Kurangi frame size (banding confidence detection threshold di config)
- Gunakan model Lite mediapipe untuk Raspberry Pi

## 6. Notes untuk Raspberry Pi Deployment

Nanti saat deploy ke Raspberry Pi 3:
- Gunakan `mediapipe-rpi` (versi ringan)
- Turunkan resolution (camera_width, camera_height)
- Increase detection confidence threshold (skip beberapa frame)
- Consider menggunakan lightweight face recognition model

## 7. Performance Monitoring

Script akan menampilkan:
- **FPS**: Frame Per Second untuk monitoring performance
- **EAR Value**: Nilai Eye Aspect Ratio real-time
- **Status Liveness**: Real/Asli atau Belum Terdeteksi
- **Recognition Result**: Nama orang / Unknown / Akses Ditolak
