<!-- portfolio -->
<!-- slug: spinach-detector -->
<!-- title: Spinach Detector -->
<!-- description: Sistem deteksi objek real-time untuk mengenali bayam menggunakan model YOLO -->
<!-- image: https://github.com/user-attachments/assets/9942fdd6-3665-4982-b59a-86ec199a0054 -->
<!-- tags: python, flask, react, vite -->

# 🥬 Spinach Detector AI

<img width="1383" height="899" alt="image" src="https://github.com/user-attachments/assets/9942fdd6-3665-4982-b59a-86ec199a0054" />

Sistem **deteksi objek real-time** untuk mengenali **bayam** menggunakan model deep learning **YOLO (You Only Look Once)**. Dibangun dengan **React** (Frontend) dan **Flask** (Backend) untuk menghadirkan proses deteksi secara langsung dengan visual bounding box dan confidence score.

---

## 👨‍💻 Developer

| Nama |
|------|
| Daffa |

---

## 🧠 Deskripsi

**Spinach Detector AI** adalah aplikasi computer vision yang memanfaatkan model YOLO modern untuk mendeteksi bayam secara real-time melalui webcam. Sistem ini menyediakan:

1. **Deteksi real-time** dengan live camera feed  
2. **Visual bounding box** bergaya YOLO (corner brackets)  
3. **Confidence score** pada setiap hasil deteksi  
4. **Dukungan multi-model** (YOLO 9 dan YOLO 11)  
5. **UI modern dan responsif** dengan efek glassmorphism  

Cocok untuk kebutuhan demonstrasi object detection maupun sebagai fondasi sistem pengenalan objek berbasis AI.

---

## ⚙️ Teknologi yang Digunakan

### Backend
- **Python 3.x**
- **Flask** – Web framework
- **Flask-CORS** – Cross-origin resource sharing
- **Ultralytics YOLO** – Model deteksi objek
- **Pillow (PIL)** – Pemrosesan gambar

### Frontend
- **React** – Framework UI
- **Vite** – Build tool & dev server
- **Axios** – HTTP client
- **CSS3** – Styling modern (gradient & glassmorphism)
- **Canvas API** – Visualisasi bounding box real-time

---

## 🚀 Cara Menjalankan Project

### Prasyarat

- Python 3.8+
- Node.js 16+ dan npm
- Akses webcam

### 1️⃣ Setup Backend (Flask)

```bash
cd backend
pip install flask flask-cors ultralytics pillow
python main.py
```

Backend akan berjalan di:
```
http://localhost:5000
```

### 2️⃣ Setup Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

Frontend akan berjalan di:
```
http://localhost:5173
```

---

## 🧩 Struktur Project

```
spinach-detector/
├── backend/
│   ├── main.py              # Server API Flask
│   ├── models/
│   │   ├── yolo9.pt         # Bobot model YOLO 9
│   │   └── yolo11.pt        # Bobot model YOLO 11
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.js           # Komponen utama React
│   │   ├── index.js         # Entry point
│   │   └── index.css        # Style global
│   ├── public/
│   └── package.json
│
└── README.md
```

---

## 🧠 Cara Kerja Sistem

1. User membuka aplikasi web dan memberikan izin kamera  
2. User memilih model YOLO (YOLO 9 atau YOLO 11)  
3. Sistem mulai menangkap frame kamera setiap 500ms  
4. Backend memproses frame menggunakan model YOLO  
5. Backend mengembalikan data deteksi:
   - Koordinat bounding box (0–1, ternormalisasi)
   - Confidence score
   - Status deteksi bayam  
6. Frontend menggambar bounding box secara real-time:
   - Warna hijau dengan corner brackets
   - Label “bayam” + persentase confidence
   - Overlay langsung di atas video

---

## 🔌 API Endpoint

### POST `/predict`

Menganalisis gambar dan mendeteksi objek bayam.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
```
image: [file gambar]
model: "yolo11" | "yolo9"
```

**Response:**
```json
{
  "is_bayam": true,
  "confidence": 95.67,
  "detections": [
    {
      "x": 0.245,
      "y": 0.312,
      "width": 0.189,
      "height": 0.234,
      "confidence": 95.67
    }
  ]
}
```

**Keterangan Field:**
- `is_bayam` → Apakah bayam terdeteksi
- `confidence` → Confidence tertinggi (0–100)
- `detections` → Daftar objek terdeteksi

---

## ✨ Fitur Utama

### 🎨 Desain UI Modern
- Efek **glassmorphism**
- Background gradient
- Dark mode
- Responsif (desktop & mobile)
- Animasi halus

### 📦 Bounding Box Real-time
- Gaya YOLO (corner brackets)
- Warna hijau terang
- Label confidence
- Multi-detection

### 📊 Dashboard Statistik
- Confidence maksimum
- Jumlah objek terdeteksi
- Model YOLO aktif

### 🎯 Kontrol Deteksi
- Ganti model YOLO
- Start / Stop deteksi
- Status indikator
- Error handling

---

## 🎮 Cara Menggunakan

1. Buka `http://localhost:5173`
2. Izinkan akses kamera
3. Pilih model YOLO:
   - YOLO 11 (disarankan, lebih akurat)
   - YOLO 9 (lebih ringan & cepat)
4. Klik **Start Detection**
5. Arahkan bayam ke kamera
6. Bounding box dan confidence akan muncul
7. Klik **Stop** untuk menghentikan

---

## 🔧 Kustomisasi

### Mengganti Objek Deteksi

Di `main.py`:
```python
if model.names[cls].lower() == "nama_objek":
```

### Mengatur Kecepatan Deteksi

Di `App.js`:
```javascript
intervalRef.current = setInterval(captureAndDetect, 500);
```

### Mengubah Warna Bounding Box

```javascript
const boxColor = "#10b981";
```

### Menambahkan Model Baru

1. Tambahkan file `.pt` ke `backend/models/`
2. Update `main.py`
3. Tambahkan opsi di `App.js`

---

## 🐛 Troubleshooting

**Kamera tidak aktif**
- Cek permission browser
- Gunakan localhost / HTTPS

**Backend error**
- Pastikan model ada
- Pastikan Flask berjalan

**Deteksi tidak muncul**
- Pencahayaan kurang
- Model belum dilatih untuk bayam

**Performa lambat**
- Kurangi FPS
- Gunakan YOLO 9
- Turunkan resolusi video

---

## 📝 Lisensi

Open source untuk keperluan edukasi.

---

## 🤝 Kontribusi

Pull request dan issue sangat diterima.

---

## 🙏 Apresiasi

- Ultralytics YOLO
- React
- Flask
- Google Fonts (Inter)

---

**Built with ❤️ menggunakan YOLO AI**
