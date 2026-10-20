# Sistem Autentikasi Pengenalan Wajah dengan Liveness Detection (EAR)
### *Teroptimasi untuk Raspberry Pi 3 (RAM 1 GB, Quad-Core CPU)*

Proyek Tugas Akhir / Skripsi yang mengimplementasikan sistem verifikasi biometrik wajah real-time dengan mekanisme anti-spoofing berbasis kedipan mata (*Eye Aspect Ratio - EAR*), dirancang khusus untuk berjalan lancar pada perangkat *single-board computer* dengan sumber daya terbatas (*edge device*).

---

## 🚀 Fitur Utama & Optimasi Raspberry Pi 3

1. **Multithreaded Camera Stream (`video_stream.py`)**
   - Pembacaan frame kamera I/O dilakukan pada *daemon thread* terpisah (`threading.Thread`).
   - Loop pemrosesan utama selalu membaca frame terbaru secara *non-blocking* (0 ms waiting time).
   - Pengaturan buffer ukuran 1 untuk mencegah penumpukan frame usang (*frame lag*).

2. **Downscaling & Preprocessing (320x240)**
   - Setiap frame secara ketat diproses pada resolusi **320x240**.
   - Memangkas jumlah piksel hingga **75%** dibandingkan resolusi standar 640x480, menghasilkan lonjakan drastis pada FPS dan penurunan drastis pada suhu CPU Raspberry Pi 3.

3. **Logika *Liveness First***
   - Nilai EAR (*Eye Aspect Ratio*) dihitung berdasarkan formula ilmiah **Soukupová & Čech (2016)**.
   - Deteksi kedipan konsekutif: memvalidasi mata tertutup selama 2–3 frame berturut-turut lalu terbuka kembali, menyaring kedipan palsu atau *noise* sensor kamera.
   - **Proses Face Recognition HANYA dieksekusi jika status liveness sudah `PASSED`**. Wajah palsu (foto cetak/layar HP yang tidak berkedip) tidak akan pernah membebani CPU dengan proses pengenalan wajah.

4. **Arsitektur Modular (Modular Design)**
   - **Landmark Detection**:
     - *Default*: **MediaPipe Face Mesh** (TFLite quantized, sangat ringan dan cepat di CPU ARM).
     - *Alternatif Skripsi*: **Dlib 68 Facial Landmarks** (dapat ditukar untuk perbandingan performa skripsi).
   - **Face Recognition**:
     - *Default*: **LBPH OpenCV** (*Local Binary Patterns Histograms* - latensi < 5 ms di Pi 3, sangat hemat memori).
     - *Deep Learning*: **MobileFaceNet TFLite** (CNN ringan untuk edge device dengan akurasi 99.5%).
     - *Alternatif Skripsi*: **Dlib ResNet 128-D** (kompatibel dengan format database lama `encodings.pickle`).

5. **Logging Metrik Skripsi & OSD**
   - Menghitung dan menampilkan di layar secara real-time: **FPS**, **Inference Time (ms)**, **CPU Usage (%)**, dan **RAM Usage (MB)** menggunakan modul `psutil` dan `time`.
   - Fasilitas ekspor metrik frame-by-frame ke format **CSV** pada folder `logs/` untuk tabel analisis bab 4 & 5 skripsi.

---

## 📐 Alur Sistem (State Machine Liveness-First)

```
[Kamera 320x240]
       │
       ▼
[Deteksi Landmark (MediaPipe / Dlib)]
       │
       ├─► Tidak Ada Wajah ──► State: NO_FACE (Standby)
       │
       ▼
[Hitung Rasio Bukaan Mata (EAR)]
       │
       ├─► Kedip Belum Valid ──► State: LIVENESS_CHECKING ("Silakan Berkedip")
       │
       ▼ (Kedipan Konsekutif Valid Terpenuhi)
[State: LIVENESS_PASSED]
       │
       ▼ (HANYA DIJALANKAN KETIKA LIVENESS SUDAH PASSED)
[Face Recognition (LBPH / MobileFaceNet / Dlib)]
       │
       ├─► Wajah Cocok ──────► State: ACCESS_GRANTED (Hijau)
       └─► Wajah Beda/Asing ──► State: ACCESS_DENIED (Merah)
```

---

## 🛠️ Panduan Instalasi di Raspberry Pi 3

### 1. Rekomendasi Sistem Operasi
Disarankan menggunakan **Raspberry Pi OS 64-bit** (Debian Bullseye atau Bookworm) agar pustaka MediaPipe dan TFLite berjalan optimal dengan instruksi ARMv8 64-bit.

### 2. Menaikkan Ukuran Swap Memory (Penting untuk Pi 3 RAM 1 GB)
Sebelum menginstal dependensi atau menjalankan proses berat, tingkatkan ukuran file swap agar Pi tidak mengalami *Out of Memory (OOM)*:
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Ubah baris: CONF_SWAPSIZE=100 menjadi CONF_SWAPSIZE=1024 atau 2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 3. Aktifkan Modul Kamera (Jika menggunakan Pi Camera Module)
```bash
sudo raspi-config
# Pilih 'Interface Options' -> Aktifkan 'Camera' / 'Legacy Camera' jika dibutuhkan
sudo reboot
```

### 4. Instal Dependensi Sistem & Python
```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv libatlas-base-dev

# Clone atau masuk ke direktori proyek
cd ~/Codes/Skripsi

# Instal seluruh modul python
pip3 install -r requirements.txt
```

Jika ingin menggunakan runtime MobileFaceNet tanpa menginstal TensorFlow penuh:
```bash
pip3 install tflite-runtime
```

---

## 💻 Cara Menjalankan Program

### 1. Menjalankan Default (MediaPipe + LBPH - Sangat Disarankan untuk Pi 3)
```bash
python3 main.py
```

### 2. Mengaktifkan Pencatatan Metrik ke File CSV (Untuk Data Pengujian Skripsi)
```bash
python3 main.py --log-csv
```
File hasil pencatatan akan otomatis tersimpan di direktori `logs/metrics_mediapipe_lbph_YYYYMMDD_HHMMSS.csv`.

### 3. Komparasi Metode untuk Bahan Analisis Skripsi
- **Menggunakan MobileFaceNet (TFLite)**:
  ```bash
  python3 main.py --detector mediapipe --recognizer mobilefacenet --log-csv
  ```
- **Menggunakan Dlib (Untuk pembanding performa CPU & FPS)**:
  ```bash
  python3 main.py --detector dlib --recognizer dlib --log-csv
  ```

### 4. Menjalankan Tanpa Antarmuka GUI (Headless Mode)
Cocok jika Raspberry Pi diakses via SSH tanpa monitor monitor fisik:
```bash
python3 main.py --no-gui --log-csv
```

---

## ⌨️ Tombol Kontrol Keyboard

| Tombol | Fungsi |
| :---: | :--- |
| `e` | **Enroll Wajah**: Mendaftarkan wajah yang sedang ada di depan kamera ke database. Ambil 3–5 sampel dengan berbagai ekspresi/sudut. |
| `r` | **Reset Status**: Mengembalikan status verifikasi ke awal (berguna saat pergantian subjek pengujian). |
| `s` | **Toggle CSV**: Menyalakan atau mematikan pencatatan metrik ke file CSV secara dinamis. |
| `q` / `ESC` | **Keluar**: Menutup koneksi kamera dan membersihkan *resources* dengan aman. |

---

## 📊 Format Data Log CSV (Metrik Skripsi)

Setiap baris pada file CSV memuat kolom:
1. `Timestamp`: Waktu pencatatan (milidetik).
2. `Frame_ID`: Nomor urutan frame.
3. `FPS`: *Frames Per Second* saat frame diproses.
4. `Landmark_Time_ms`: Waktu komputasi deteksi landmark wajah (ms).
5. `Recog_Time_ms`: Waktu komputasi pengenalan wajah (ms, hanya terisi saat tahap pengenalan aktif).
6. `Total_Pipeline_ms`: Total waktu latensi satu iterasi frame (ms).
7. `Process_CPU_Percent`: Persentase penggunaan CPU oleh program (%).
8. `Process_RAM_MB`: Konsumsi memori RAM aktual yang digunakan oleh program (MB).
9. `System_RAM_Percent`: Persentase total RAM Raspberry Pi yang terpakai (%).
10. `EAR_Value`: Nilai *Eye Aspect Ratio* mata subjek saat ini.
11. `Consecutive_Closed`: Jumlah frame berturut-turut di mana mata terpejam.
12. `Blink_Count`: Akumulasi kedipan sah yang tercatat.
13. `Liveness_Status`: State status (`NO_FACE`, `LIVENESS_CHECKING`, `LIVENESS_PASSED`, `ACCESS_GRANTED`, `ACCESS_DENIED`).
14. `Recognized_Name`: Identitas subjek yang dikenali (`Hamzah`, `Unknown`, dll.).
15. `Distance_Score`: Nilai jarak biometrik kemiripan fitur wajah.
