---
title: BISINDO.AI — Indonesian Sign Language Translator
emoji: 🤟
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🤟 BISINDO.AI — Penerjemah Bahasa Isyarat Indonesia

> Deteksi huruf BISINDO (A–Z) secara real-time menggunakan  
> **MediaPipe Hand Landmarker** + **Bidirectional LSTM**

## Cara Pakai

1. Upload foto gestur tangan **atau** aktifkan webcam
2. Klik **Deteksi Huruf**
3. Lihat hasil prediksi + visualisasi 21 titik landmark
4. Gunakan **Sentence Builder** untuk menyusun kalimat

## Struktur File

```
bisindo-gradio/
├── app.py                  # Gradio app utama
├── requirements.txt
├── bisindo_model.keras     # Model BiLSTM (upload manual)
├── bisindo_assets.json     # Class names & config
└── README.md
```

> `hand_landmarker.task` (~8 MB) diunduh otomatis saat pertama kali app dijalankan.

## Model

| Metrik | Nilai |
|---|---|
| Arsitektur | Bidirectional LSTM (3 layer) |
| Kelas | 26 huruf A–Z |
| Dataset | 5.735 gambar |
| Test Accuracy | ~93% |

## Limitasi

- Hanya mendukung satu tangan per frame
- Huruf dua tangan (B) memiliki akurasi lebih rendah
- Performa optimal pada pencahayaan baik & latar belakang polos

## Author

**Faris Ahmad Rizky Azizy** — [@farisazizy289](https://github.com/farisazizy289)

## Lisensi

MIT License — bebas digunakan untuk keperluan edukasi dan non-komersial.