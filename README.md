# 🤟 BISINDO.AI — Indonesian Sign Language Translator

> Real-time Indonesian Sign Language (BISINDO) recognition using MediaPipe Hand Landmarker + Bidirectional LSTM

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?style=flat-square&logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)

**Live Demo:** [bisindo-translator.streamlit.app](https://bisindo-translator.streamlit.app)

---

## 🌏 Latar Belakang

Lebih dari **2.5 juta penyandang disabilitas pendengaran** di Indonesia menggunakan BISINDO (Bahasa Isyarat Indonesia) sebagai bahasa utama mereka. Namun sangat sedikit orang di luar komunitas tuli yang memahaminya — menciptakan gap komunikasi yang nyata dalam kehidupan sehari-hari.

**BISINDO.AI** hadir sebagai jembatan komunikasi: siapapun bisa mengarahkan kamera ke gestur tangan dan mendapatkan terjemahan huruf BISINDO secara instan, tanpa hardware khusus.

---

## ✨ Fitur

- 📸 **Upload Foto** — upload gambar gestur tangan untuk dianalisis
- 🎥 **Webcam Real-time** — deteksi gestur langsung dari kamera
- ⏱️ **Timer** — countdown sebelum capture untuk gestur 2 tangan
- ✋ **Visualisasi Landmark** — tampilan 21 titik tangan yang terdeteksi
- 📝 **Builder Kalimat** — susun huruf menjadi kata/kalimat
- ⚠️ **Confidence Threshold** — peringatan jika gestur tidak dikenali dengan yakin

---

## 🧠 Arsitektur & Pipeline

```
Input Gambar / Webcam
        ↓
MediaPipe Hand Landmarker
(Palm Detection → 21 Landmark Points)
        ↓
Feature Extraction (21 × 3 koordinat = 63 fitur)
        ↓
Normalisasi Relatif ke Wrist
        ↓
Sequence Augmentation (10 frames + Gaussian noise)
        ↓
Bidirectional LSTM (3 layer)
        ↓
Prediksi Huruf BISINDO A–Z
```

### Kenapa MediaPipe + LSTM, bukan CNN langsung?

| | CNN dari Pixel | MediaPipe + LSTM |
|---|---|---|
| Input size | 300×300×3 = 270,000 nilai | 63 koordinat saja |
| Kebutuhan GPU | Tinggi | Jalan di CPU biasa |
| Sensitivitas background | Tinggi | Invariant |
| Tangkap pola temporal | ❌ | ✅ |
| Cocok untuk real-time | Lambat | Cepat |

---

## 📊 Hasil Model

| Metrik | Nilai |
|---|---|
| Arsitektur | Bidirectional LSTM (3 layer) |
| Jumlah Kelas | 26 huruf A–Z |
| Dataset | 5,735 gambar |
| Feature Extractor | MediaPipe Hand Landmarker |

### Akurasi per Huruf (highlight)

| Huruf | Akurasi | Huruf | Akurasi |
|---|---|---|---|
| V, L, U, G, I, J, R | 100% | S | 78% |
| F, C, W, Y, X, O | ~95% | H | 77% |
| T, K, Q, N | ~86% | D | 71% |
| A, P, E, M | ~82% | B | 38%* |

> *B lebih rendah karena gesturnya melibatkan 2 tangan — model saat ini hanya memproses 1 tangan.

---

## 🗂️ Struktur Project

```
bisindo-translator/
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
└── README.md            # Dokumentasi ini
```

> **Model files** (`bisindo_model.keras`, `bisindo_assets.json`, `hand_landmarker.task`)
> diunduh **otomatis dari Google Drive** saat pertama kali app dijalankan.
> File-file ini tidak disimpan di GitHub karena ukurannya besar.

---

## 🚀 Cara Menjalankan

### Lokal

```bash
# Clone repo
git clone https://github.com/farisazizy289/bisindo-translator.git
cd bisindo-sign-language-translator

# Install dependencies
pip install -r requirements.txt
pip install opencv-python   # pakai ini untuk lokal (bukan headless)

# Jalankan
streamlit run app.py
```

> Saat pertama kali dijalankan, app akan otomatis download model dari Google Drive (~5MB).

### Streamlit Cloud

1. Fork repo ini ke akun GitHub kamu
2. Buka **[share.streamlit.io](https://share.streamlit.io)**
3. Klik **"New app"**
4. Pilih repo → set **Main file path: `app.py`**
5. Klik **"Deploy"**

---

## 📦 Dataset

**Indonesian Hand Sign Language (BISINDO) Dataset**
- Source: [Kaggle — kelsha](https://www.kaggle.com/datasets/kelsha/indonesian-hand-sign-language-bisindo-dataset)
- Total: 5,735 gambar (4,796 train / 939 val)
- Kelas: 26 huruf A–Z
- Format: YOLO (images + labels terpisah)
- Distribusi: Seimbang ~220 gambar per kelas

---

## ⚙️ Training Pipeline

Training dilakukan di **Kaggle Notebook** dengan GPU T4:

```
1. Load dataset YOLO → konversi ke DataFrame
2. Ekstraksi 21 landmark via MediaPipe Hand Landmarker
3. Normalisasi koordinat relatif ke wrist (position invariant)
4. Augmentasi temporal (sequence 10 frames + Gaussian noise σ=0.01)
5. Label encoding (A=0, B=1, ..., Z=25)
6. Train/Val/Test split 80/10/10
7. Training Bidirectional LSTM (max 80 epochs, early stopping patience=10)
8. Evaluasi & simpan best model (monitor: val_accuracy)
```

Notebook lengkap: [`BISINDO_SignLanguage_Translator.ipynb`](./BISINDO_SignLanguage_Translator.ipynb)

---

## ⚠️ Limitasi & Etika

- Model hanya mencakup abjad A–Z, belum termasuk angka 0–9 atau kata
- Huruf B memiliki akurasi lebih rendah karena melibatkan kedua tangan
- Performa optimal pada kondisi pencahayaan baik dengan latar belakang polos
- Tidak diuji pada kondisi ekstrem (cahaya sangat redup, tangan sangat kecil di frame)
- **Aplikasi ini adalah alat bantu komunikasi, bukan pengganti interpreter BISINDO profesional**
- Tidak dirancang untuk penggunaan medis atau legal

---

## 🔮 Pengembangan Selanjutnya

- [ ] Tambah angka 0–9
- [ ] Support gestur 2 tangan (perbaiki akurasi huruf B)
- [ ] Perluas ke kata dan frasa umum BISINDO
- [ ] Integrasi text-to-speech untuk output suara
- [ ] Mobile app dengan Core ML (iOS/iPadOS)
- [ ] Dataset augmentasi dengan variasi pencahayaan

---

## 👤 Author

**Faris Azizy**
- GitHub: [@farisazizy289](https://github.com/farisazizy289)
- Portfolio: Apple Developer Academy Indonesia Applicant

---

## 📄 Lisensi

MIT License — bebas digunakan untuk keperluan edukasi dan non-komersial.

---

> *"Technology is most powerful when it empowers everyone."* — Tim Cook, Apple
>
> BISINDO.AI dibangun dengan semangat bahwa teknologi harus inklusif untuk semua —
> termasuk 2.5 juta penyandang disabilitas pendengaran di Indonesia.
