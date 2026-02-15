---
title: Churn Prediction API
emoji: 📉
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# Proyek Prediksi Churn Pengguna

Proyek ini bertujuan untuk memprediksi apakah seorang pengguna akan berhenti menggunakan aplikasi (churn) atau tetap aktif berdasarkan pola penggunaan mereka.

## Fitur yang Digunakan
### 1. Informasi Identitas (Hanya untuk referensi):
- **username**: Nama unik pengguna.
- **email**: Alamat email pengguna.
- **no_wa**: Nomor WhatsApp pengguna.

### 2. Fitur Perilaku (Digunakan untuk Prediksi):
- **login_freq**: Seberapa sering pengguna login dalam sebulan.
- **last_login_days**: Jumlah hari sejak login terakhir.
- **total_transactions**: Total transaksi yang dilakukan pengguna.
- **avg_session_time**: Rata-rata waktu yang dihabiskan dalam aplikasi (menit).

## Teknologi yang Digunakan
- **Python** (Machine Learning & API)
- **Scikit-Learn** (Logistic Regression)
- **FastAPI** (REST API)
- **Joblib** (Model Serialization)
- **Docker** (Containerization for Deployment)

## Cara Menjalankan Secara Lokal
### 1. Training & Evaluasi Model
```bash
pip install -r requirements.txt
python churn_prediction.py
```

### 2. Menjalankan REST API (FastAPI)
```bash
uvicorn main:app --reload
```
Aplikasi akan berjalan di `http:localhost:8000`. Dokumentasi di `/docs`.

## Keamanan API (API Key)
API ini dilindungi menggunakan API Key guna memastikan hanya Anda yang bisa mengaksesnya.
- **Nama Header**: `X-API-KEY`
- **Isi Header**: (Sesuai dengan `API_KEY` di file `.env` atau Secret di Hugging Face)

## Deployment Online (Hugging Face Spaces)
1. **Buat Space baru** di Hugging Face dengan tipe **Docker (Blank)**.
2. **Unggah file**: `main.py`, `requirements.txt`, `Dockerfile`, `churn_model.joblib`, `scaler.joblib`.
3. **Atur Secret**: Di tab Settings, tambahkan secret `API_KEY` dengan kunci rahasia Anda.

---
## Output Proyek
- `user_churn_data.csv`: Dataset sintetis.
- `churn_model.joblib`: Model terlatih.
- `scaler.joblib`: Objek normalisasi.
- `confusion_matrix.png`: Metrik performa.
- `feature_importance.png`: Fitur paling berpengaruh.
