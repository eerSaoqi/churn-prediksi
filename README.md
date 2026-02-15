# User Churn Prediction Project

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
- **Python** sebagai bahasa pemrograman utama.
- **Scikit-Learn** untuk machine learning (Logistic Regression).
- **Pandas & Numpy** untuk pengolahan data.
- **Matplotlib & Seaborn** untuk visualisasi data.
- **Joblib** untuk menyimpan model yang telah dilatih.

## Cara Menjalankan
### 1. Training & Evaluasi Model
```bash
pip install -r requirements.txt
python churn_prediction.py
```

### 2. Menjalankan REST API (FastAPI)
```bash
uvicorn main:app --reload
```
Aplikasi akan berjalan di `http:localhost:8000`.

## Dokumentasi API
FastAPI menyediakan dokumentasi otomatis yang bisa diakses di:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoint `POST /predict`
**Contoh Payload Input:**
```json
{
  "user_id": "USER-001",
  "login_freq": 5,
  "last_login_days": 40,
  "total_transactions": 2,
  "avg_session_time": 10.5
}
```

**Contoh Response:**
```json
{
  "user_id": "USER-001",
  "churn_probability": 0.9982,
  "risk_level": "HIGH"
}
```

## Keamanan API (API Key)
API ini dilindungi menggunakan API Key guna memastikan hanya Anda yang bisa mengaksesnya.
- **Nama Header**: `X-API-KEY`
- **Isi Header**: (Sesuai dengan `API_KEY` di file `.env`)

Jika menggunakan `curl`, tambahkan header seperti ini:
```bash
-H "X-API-KEY: RAHASIA_SAYA_123"
```

## Deployment Online (Render.com)

1.  **Push ke GitHub**:
    - Buat repositori baru di GitHub.
    - Push semua file kecuali yang ada di `.gitignore` (file model `.joblib` harus ikut di-push).
2.  **Daftar Render.com**:
    - Pilih **New > Web Service**.
    - Hubungkan dengan repositori GitHub Anda.
3.  **Konfigurasi di Render**:
    - **Runtime**: `Python 3`
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4.  **Atur Environment Variables** (Sangat Penting):
    - Masuk ke tab **Environment** di Render.
    - Tambahkan: `API_KEY` = `KunciRahasiaAnda` (isi sesuai keinginan Anda).

---

## Output Proyek
- `user_churn_data.csv`: Dataset sintetis hasil simulasi.
- `churn_model.joblib`: Model Logistic Regression.
- `scaler.joblib`: Objek scaler untuk normalisasi data.
- `confusion_matrix.png`: Performa model.
- `feature_importance.png`: Fitur paling berpengaruh.
