# 🔧 Troubleshooting: MediaPipe AttributeError

## Error Yang Anda Alami

```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

---

## ✅ SOLUTION CEPAT (2 Menit)

### Option 1: Jalankan Fix Script (RECOMMENDED)

```bash
python fix_mediapipe.py
```

Script ini akan:
1. ✓ Uninstall MediaPipe yang current
2. ✓ Install versi yang benar (0.8.11)
3. ✓ Verify installation
4. ✓ Install dependencies lainnya

Setelah selesai, jalankan:
```bash
python main.py
```

---

## Option 2: Manual Fix (Jika fix_mediapipe.py tidak bekerja)

### Step 1: Uninstall MediaPipe
```bash
pip uninstall -y mediapipe opencv-contrib-python
```

### Step 2: Upgrade pip
```bash
python -m pip install --upgrade pip
```

### Step 3: Install dependencies yang benar
```bash
pip install numpy>=1.19.0,<1.25.0
pip install opencv-python>=4.5.0,<4.9.0
pip install mediapipe==0.8.11
pip install face-recognition pillow scikit-image
```

### Step 4: Verify installation
```bash
python -c "import mediapipe as mp; print(mp.solutions.face_mesh); print('✓ OK')"
```

Jika output: `✓ OK`, maka berhasil!

### Step 5: Run program
```bash
python main.py
```

---

## Option 3: Gunakan Fallback (OpenCV Mode)

Jika MediaPipe tetap error, gunakan versi dengan fallback:

```bash
python main_with_fallback.py
```

Script ini akan:
- ✓ Coba MediaPipe dulu
- ✓ Jika gagal, automatic fallback ke OpenCV Cascade
- ✓ Program tetap jalan, hanya akurasi sedikit lebih rendah

---

## 🔍 Penyebab Error

Biasanya error terjadi karena:

1. **MediaPipe tidak ter-install**
   - Solution: `pip install mediapipe==0.8.11`

2. **Versi MediaPipe yang tidak kompatibel**
   - Beberapa versi media pipe memiliki structure berbeda
   - Solution: Install versi spesifik 0.8.11

3. **Conflict dengan package lain**
   - Terutama opencv-contrib-python vs opencv-python
   - Solution: `pip uninstall opencv-contrib-python` kemudian reinstall

4. **Environment atau Python version issue**
   - Python 3.7-3.10 direkomendasikan
   - Python 3.11+ mungkin belum fully supported

---

## 📋 Verification Checklist

Setelah fix, check berikut:

### 1. Verify MediaPipe Import
```bash
python -c "import mediapipe as mp; print(mp.__version__)"
```

Expected output: `0.8.11` atau versi sejenis

### 2. Verify Face Mesh
```bash
python -c "import mediapipe as mp; fm = mp.solutions.face_mesh; print('✓ Face Mesh OK')"
```

Expected output: `✓ Face Mesh OK`

### 3. Verify Face Recognition
```bash
python -c "import face_recognition; print('✓ Face Recognition OK')"
```

Expected output: `✓ Face Recognition OK`

### 4. Verify OpenCV
```bash
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

Expected output: `OpenCV: 4.X.X` (versi 4.5.0+)

---

## 🛠️ If Everything Still Fails

### Create New Virtual Environment

Kadang issue ada di environment. Buat environment baru:

```bash
# Backup old environment (optional)
# mv venv venv_backup

# Create new virtual environment
python -m venv venv_new

# Activate
# Windows:
venv_new\Scripts\activate
# Linux/Mac:
source venv_new/bin/activate

# Install fresh
pip install --upgrade pip
pip install -r requirements.txt

# Test
python main.py
```

---

## 📞 If Still Having Issues

### Debug Mode - Print Detailed Info

```bash
python -c "
import sys
print('Python version:', sys.version)
print()

try:
    import mediapipe
    print('MediaPipe:', mediapipe.__file__)
    print('Has solutions:', hasattr(mediapipe, 'solutions'))
    if hasattr(mediapipe, 'solutions'):
        print('Has face_mesh:', hasattr(mediapipe.solutions, 'face_mesh'))
except Exception as e:
    print('ERROR:', e)

print()

try:
    import cv2
    print('OpenCV:', cv2.__version__)
except Exception as e:
    print('ERROR:', e)

print()

try:
    import face_recognition
    print('Face Recognition: OK')
except Exception as e:
    print('ERROR:', e)
"
```

Simpan output untuk reference.

---

## 🎯 Recommended Solution Path

**Untuk 95% kasus, ini yang work:**

```bash
# 1. Create/activate virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Uninstall problematic packages
pip uninstall -y mediapipe opencv-contrib-python

# 4. Install requirements
pip install -r requirements.txt

# 5. Run
python main.py
```

---

## 💡 Pro Tips

### 1. Gunakan Conda (Alternative)

Jika pip terus bermasalah, coba conda:

```bash
conda create -n skripsi python=3.9
conda activate skripsi
conda install -c conda-forge mediapipe
pip install face-recognition
python main.py
```

### 2. Docker (Ultimate Solution)

Jika environment masih bermasalah, gunakan Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
```

```bash
docker build -t skripsi .
docker run --device /dev/video0 -e DISPLAY=$DISPLAY skripsi
```

### 3. Use Google Colab (Quick Test)

Jika laptop bermasalah, test di Google Colab:

```python
!pip install mediapipe face-recognition
!python main.py
```

---

## 📚 Reference

- MediaPipe: https://github.com/google/mediapipe
- Face Recognition: https://github.com/ageitgey/face_recognition
- OpenCV: https://opencv.org

---

## ✅ Success Criteria

Setelah fix, Anda seharusnya bisa:

- [ ] Run `python main.py` tanpa error
- [ ] Webcam terdeteksi dan show live feed
- [ ] Face detection berfungsi (ada bounding box saat ada wajah)
- [ ] EAR values ditampilkan
- [ ] Blink terdeteksi
- [ ] Face recognition berfungsi

---

**Jika masih error setelah semua ini, gunakan fallback:**

```bash
python main_with_fallback.py
```

Script ini akan fallback ke OpenCV jika MediaPipe gagal, sehingga program tetap jalan! 🎉

