from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from contextlib import asynccontextmanager
import pandas as pd
from dotenv import load_dotenv

# --- 1. Load Environment Variables ---
load_dotenv()
API_KEY = os.getenv("API_KEY") 

if not API_KEY:
    # Jika tidak ada di .env, gunakan default tapi berikan peringatan
    print("WARNING: API_KEY not found in .env, using default development key.")
    API_KEY = "RAHASIA_SAYA_123"

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Definisikan nama fitur agar sesuai saat training (untuk menghindari UserWarning)
FEATURE_NAMES = ['login_freq', 'last_login_days', 'total_transactions', 'avg_session_time']

# --- 2. Global Variables for Model ---
model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    try:
        model = joblib.load('churn_model.joblib')
        scaler = joblib.load('scaler.joblib')
        print(f"--- Model and Scaler loaded! Security enabled with {API_KEY_NAME} ---")
    except Exception as e:
        print(f"Error loading assets: {e}")
        raise RuntimeError("Failed to load model assets")
    yield

# --- 3. Inisialisasi FastAPI ---
app = FastAPI(
    title="Churn Prediction API (Secure)",
    lifespan=lifespan
)

# --- 4. Middleware CORS (Penting agar dashboard bisa akses API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ganti dengan domain dashboard Anda nantinya untuk lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. Security Dependency ---
async def get_api_key(header_key: str = Security(api_key_header)):
    if header_key == API_KEY:
        return header_key
    raise HTTPException(
        status_code=403, 
        detail="Akses Ditolak: API Key tidak valid atau tidak ada"
    )

# --- 6. Pydantic Models ---
class PredictionRequest(BaseModel):
    user_id: str
    login_freq: int = Field(..., ge=0)
    last_login_days: int = Field(..., ge=0)
    total_transactions: int = Field(..., ge=0)
    avg_session_time: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    user_id: str
    churn_probability: float
    risk_level: str

# --- 7. Helper ---
def get_risk_level(prob: float) -> str:
    if prob < 0.4: return "LOW"
    elif 0.4 <= prob <= 0.7: return "MEDIUM"
    else: return "HIGH"

# --- 8. Endpoints ---
@app.get("/")
async def root():
    return {"message": "Secure Churn Prediction API is running!"}

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(get_api_key)])
async def predict(data: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocessing: Gunakan DataFrame agar nama fitur sesuai dengan saat training
        # Ini akan menghilangkan "UserWarning: X does not have valid feature names"
        input_data = pd.DataFrame([[
            data.login_freq, 
            data.last_login_days, 
            data.total_transactions, 
            data.avg_session_time
        ]], columns=FEATURE_NAMES)
        
        scaled_features = scaler.transform(input_data)
        
        # Prediction
        prob = model.predict_proba(scaled_features)[0][1]
        
        return PredictionResponse(
            user_id=data.user_id,
            churn_probability=round(float(prob), 4),
            risk_level=get_risk_level(prob)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
