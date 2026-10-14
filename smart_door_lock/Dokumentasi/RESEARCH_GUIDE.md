# Research Guide - Membandingkan Model Algoritma

Panduan ini menjelaskan bagaimana cara menggunakan framework ini untuk membandingkan berbagai model algoritma untuk skripsi Anda.

---

## 📐 Arsitektur yang Modular

Sistem ini dirancang dengan **Object-Oriented Programming (OOP)** dan **separation of concerns** sehingga mudah untuk:
- Swap model algoritma
- Membandingkan performance
- Menambah fitur baru

### Tiga Komponen Utama yang Bisa di-Swap

```
Face Detection (MediaPipe)      ──┐
                                  ├──> PIPELINE ──> Display & Logging
Liveness Detection (EAR)        ──┤
                                  │
Face Recognition (face_recognition) ┘
```

Setiap komponen di-wrap dalam class terpisah yang bisa diganti.

---

## 🔄 Cara 1: Swap Face Detection Algorithm

### Current Implementation
File: `modules/face_detector.py`
- Menggunakan: **MediaPipe Face Mesh**
- Method: Detect 468 facial landmarks

### Alternatif yang Bisa Dicoba

**Opsi 1: OpenCV DNN Face Detector**
```python
# modules/face_detector_opencv.py (new file)

class FaceDetectorOpenCV:
    def __init__(self):
        model_file = "res10_300x300_ssd_iter_140000.caffemodel"
        proto_file = "deploy.prototxt.txt"
        self.net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
    
    def detect_faces(self, frame):
        # Implementation with OpenCV DNN
        pass
```

**Opsi 2: YOLO Face Detector**
```python
# modules/face_detector_yolo.py (new file)

class FaceDetectorYOLO:
    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet("yoloface.cfg", "yoloface.weights")
    
    def detect_faces(self, frame):
        # Implementation with YOLO
        pass
```

**Opsi 3: RetinaFace**
```python
# modules/face_detector_retinaface.py (new file)

class FaceDetectorRetinaFace:
    def __init__(self):
        from retinaface import RetinaFace
        self.detector = RetinaFace.build_model("resnet50")
    
    def detect_faces(self, frame):
        # Implementation with RetinaFace
        pass
```

### Cara Menggunakan Alternatif
```python
# main.py - SmartDoorLockSystem.__init__()

# OPTION A: MediaPipe (current)
from modules.face_detector import FaceDetector
self.face_detector = FaceDetector()

# OPTION B: OpenCV
from modules.face_detector_opencv import FaceDetectorOpenCV
self.face_detector = FaceDetectorOpenCV()

# OPTION C: YOLO
from modules.face_detector_yolo import FaceDetectorYOLO
self.face_detector = FaceDetectorYOLO()
```

### Perlu Update
- Signature method `detect_faces()` harus sama:
  ```python
  def detect_faces(self, frame) -> Tuple[List[dict], np.ndarray]:
      # Returns: (faces_list, annotated_frame)
  ```

---

## 👁️ Cara 2: Swap Liveness Detection Algorithm

### Current Implementation
File: `modules/liveness_detector.py`
- Method: **Eye Aspect Ratio (EAR)**
- Parameter yang di-monitor: Tinggi/lebar mata

### Alternatif yang Bisa Dicoba

**Opsi 1: Texture Analysis (LBP)**
```python
# modules/liveness_detector_texture.py

class LivenessDetectorTexture:
    def __init__(self):
        pass
    
    def detect_liveness(self, face_region):
        # Analisis texture menggunakan LBP (Local Binary Patterns)
        # Real face vs spoofing/presentation attack
        from skimage.feature import local_binary_pattern
        
        lbp = local_binary_pattern(face_region, P=8, R=1)
        # Extract histogram dan bandingkan
        pass
```

**Opsi 2: Frequency Analysis (Fourier)**
```python
# modules/liveness_detector_frequency.py

class LivenessDetectorFrequency:
    def detect_liveness(self, face_region):
        # Fourier Transform analysis
        # Real faces dan spoofing memiliki frequency signature berbeda
        import numpy as np
        
        fft = np.fft.fft2(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
        magnitude = np.abs(fft)
        # Compare frequency pattern
        pass
```

**Opsi 3: Multi-Spectral Analysis**
```python
# modules/liveness_detector_multispectral.py

class LivenessDetectorMultiSpectral:
    def detect_liveness(self, face_region):
        # Analisis RGB channels separately
        # Real faces vs paper/screen memiliki behavior berbeda
        pass
```

**Opsi 4: Deep Learning (CNN)**
```python
# modules/liveness_detector_dl.py

import tensorflow as tf

class LivenessDetectorDL:
    def __init__(self):
        # Load pre-trained model
        self.model = tf.keras.models.load_model("liveness_model.h5")
    
    def detect_liveness(self, face_region):
        # Preprocessing
        x = cv2.resize(face_region, (224, 224))
        x = x.astype('float32') / 255.0
        x = np.expand_dims(x, axis=0)
        
        # Predict
        prediction = self.model.predict(x)
        is_alive = prediction[0][0] > 0.5
        confidence = prediction[0][0]
        
        return is_alive, confidence
```

### Cara Menggunakan Alternatif
```python
# main.py

# OPTION A: EAR (current)
from modules.liveness_detector import LivenessDetector
self.liveness_detector = LivenessDetector()

# OPTION B: Texture Analysis
from modules.liveness_detector_texture import LivenessDetectorTexture
self.liveness_detector = LivenessDetectorTexture()

# OPTION C: Deep Learning
from modules.liveness_detector_dl import LivenessDetectorDL
self.liveness_detector = LivenessDetectorDL()
```

### Perlu Update
- Method untuk liveness detection:
  ```python
  def detect_liveness(self, face_region) -> bool:
      # Returns: True jika real, False jika spoofing
      pass
  ```

---

## 👤 Cara 3: Swap Face Recognition Algorithm

### Current Implementation
File: `modules/face_recognizer.py`
- Model: **face_recognition (dlib)**
- Encoding: 128-dimensional vector
- Distance metric: Euclidean distance

### Alternatif yang Bisa Dicoba

**Opsi 1: VGGFace2**
```python
# modules/face_recognizer_vgg.py

from keras_vggface.vggface import VGGFace

class FaceRecognizerVGG:
    def __init__(self):
        self.model = VGGFace(model='vgg16')
    
    def encode_face(self, face_image):
        # VGGFace encoding
        x = cv2.resize(face_image, (224, 224))
        x = np.expand_dims(x, axis=0)
        encoding = self.model.predict(x)
        return encoding
```

**Opsi 2: FaceNet**
```python
# modules/face_recognizer_facenet.py

class FaceRecognizerFaceNet:
    def __init__(self):
        from facenet_pytorch import InceptionResnetV1
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
    
    def encode_face(self, face_image):
        # FaceNet embedding
        pass
```

**Opsi 3: ArcFace**
```python
# modules/face_recognizer_arcface.py

class FaceRecognizerArcFace:
    def __init__(self):
        # Load ArcFace model
        pass
    
    def encode_face(self, face_image):
        # ArcFace embedding (512-d)
        pass
```

**Opsi 4: DeepFace**
```python
# modules/face_recognizer_deepface.py

from deepface import DeepFace

class FaceRecognizerDeepFace:
    def encode_face(self, face_image):
        embedding = DeepFace.represent(face_image, model_name="VGG-Face")
        return embedding
```

### Cara Menggunakan Alternatif
```python
# main.py

# OPTION A: face_recognition (current)
from modules.face_recognizer import FaceRecognizer
self.face_recognizer = FaceRecognizer()

# OPTION B: VGGFace2
from modules.face_recognizer_vgg import FaceRecognizerVGG
self.face_recognizer = FaceRecognizerVGG()

# OPTION C: ArcFace
from modules.face_recognizer_arcface import FaceRecognizerArcFace
self.face_recognizer = FaceRecognizerArcFace()
```

---

## 📊 Cara 4: Membandingkan Multiple Models

Untuk membandingkan beberapa model sekaligus:

```python
# compare_models.py

import cv2
import time
from modules import FaceDetector
from modules.face_detector_opencv import FaceDetectorOpenCV
from modules.face_detector_yolo import FaceDetectorYOLO

class ModelComparator:
    def __init__(self):
        self.detectors = {
            'MediaPipe': FaceDetector(),
            'OpenCV DNN': FaceDetectorOpenCV(),
            'YOLO': FaceDetectorYOLO()
        }
        
        self.results = {name: [] for name in self.detectors.keys()}
    
    def compare(self, frame):
        """Compare semua model pada frame yang sama."""
        
        for model_name, detector in self.detectors.items():
            start_time = time.time()
            faces, _ = detector.detect_faces(frame)
            elapsed_time = time.time() - start_time
            
            self.results[model_name].append({
                'num_faces': len(faces),
                'time_ms': elapsed_time * 1000
            })
    
    def print_stats(self):
        """Print statistics."""
        
        for model_name, results in self.results.items():
            times = [r['time_ms'] for r in results]
            avg_time = sum(times) / len(times)
            print(f"{model_name}: {avg_time:.2f}ms average")


# Usage
comparator = ModelComparator()

camera = cv2.VideoCapture(0)
for _ in range(100):  # 100 frames
    ret, frame = camera.read()
    if ret:
        comparator.compare(frame)

comparator.print_stats()
```

---

## 📈 Metrics untuk Evaluasi

### Performance Metrics
```python
# metrics.py

import time
from collections import defaultdict

class PerformanceMetrics:
    def __init__(self):
        self.times = defaultdict(list)
        self.results = defaultdict(list)
    
    def record_time(self, component, elapsed_time):
        """Record execution time untuk component."""
        self.times[component].append(elapsed_time)
    
    def record_result(self, component, result):
        """Record hasil detection."""
        self.results[component].append(result)
    
    def get_average_time(self, component):
        """Get average time untuk component."""
        if not self.times[component]:
            return 0
        return sum(self.times[component]) / len(self.times[component])
    
    def get_fps(self, component):
        """Get FPS untuk component."""
        avg_time = self.get_average_time(component)
        if avg_time > 0:
            return 1000 / avg_time
        return 0
    
    def print_report(self):
        """Print performance report."""
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        for component in self.times.keys():
            avg_time = self.get_average_time(component)
            fps = self.get_fps(component)
            print(f"\n{component}:")
            print(f"  Average Time: {avg_time:.2f}ms")
            print(f"  FPS: {fps:.2f}")
```

### Accuracy Metrics
```python
# Untuk face recognition accuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_face_recognizer(recognizer, test_encodings, test_labels):
    """Evaluate face recognizer accuracy."""
    
    predictions = []
    for encoding in test_encodings:
        person_name, confidence = recognizer.recognize_face(encoding)
        predictions.append(person_name)
    
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-Score: {f1:.2%}")
```

---

## 🧪 Testing Strategy untuk Penelitian

### Test Scenario 1: Different Lighting Conditions
```python
# Test dengan berbagai kondisi pencahayaan
scenarios = [
    'bright_light',
    'normal_light',
    'dim_light',
    'low_light'
]
```

### Test Scenario 2: Different Angles
```python
# Test dengan berbagai angle
angles = [
    'frontal',      # 0 derajat
    'left_30',      # 30 derajat ke kiri
    'left_45',      # 45 derajat ke kiri
    'right_30',     # 30 derajat ke kanan
    'right_45'      # 45 derajat ke kanan
]
```

### Test Scenario 3: Liveness Attacks (Spoofing)
```python
# Test dengan spoofing attempts
attacks = [
    'print_photo',      # Foto fisik
    'screen_display',   # Display di layar
    'video_playback',   # Video playback
    'mask_attack'       # Wajah palsu
]
```

---

## 💾 Data Collection untuk Evaluation

### Struktur Dataset
```
evaluation_dataset/
├── ground_truth.csv         # CSV dengan format: image_path, person_name, condition, angle, attack_type
├── images/
│   ├── person_1/
│   │   ├── frontal_normal.jpg
│   │   ├── left_45_dim.jpg
│   │   └── ...
│   └── person_2/
│       └── ...
└── results/
    ├── model_a_results.csv
    ├── model_b_results.csv
    └── comparison_report.txt
```

---

## 🎓 Untuk Skripsi Anda

### Suggested Comparison Structure

**Chapter: Perbandingan Model**

Misalnya membandingkan 3 liveness detection method:
1. **Eye Aspect Ratio (EAR)** - Current implementation
2. **Texture Analysis (LBP)**
3. **Deep Learning (CNN)**

Setiap method akan dievaluasi berdasarkan:
- **Accuracy**: TPR, FPR, accuracy rate
- **Performance**: Execution time, FPS
- **Robustness**: Terhadap variasi pencahayaan, angle, attack type

---

## 📝 Dokumentasi untuk Laporan

Saat menulis laporan, dokumentasikan:

1. **Implementasi Detail**
   - Library dan versi yang digunakan
   - Parameter konfigurasi
   - Training data (jika ada)

2. **Metodologi Testing**
   - Dataset yang digunakan
   - Kondisi testing
   - Metrics yang digunakan

3. **Hasil dan Analisis**
   - Tabel perbandingan
   - Graph/chart visualization
   - Diskusi kelebihan dan kekurangan

4. **Rekomendasi**
   - Model mana yang terbaik untuk deployment
   - Trade-off antara accuracy vs performance
   - Saran untuk improvement

---

## 🔗 Reference

### Papers untuk Citation

**Face Detection:**
- Tereza Soukupová et al. "Real-Time Eye Blink Detection using Facial Landmarks" (2016)
- Kaiming He et al. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (2016)

**Liveness Detection:**
- Ioannis Komogortsev et al. "Liveness Detection via Pupil Tracking" (2015)
- Yaojie Liu et al. "FaceGuard: Protecting Personal Data Against Unauthorized Face Detection and Analysis" (2020)

**Face Recognition:**
- David G. Lowe. "Distinctive Image Features from Scale-Invariant Keypoints" (2004)
- Florian Schroff et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering" (2015)

---

Happy researching dan semoga skripsi sukses! 🎓✨
