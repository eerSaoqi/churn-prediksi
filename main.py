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
from supabase import create_client, Client

# --- 1. Load Environment Variables ---
load_dotenv()
API_KEY = os.getenv("API_KEY") 
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY not found in environment variables. Please set it in .env or secrets.")

# Initialize Supabase Client (Optional - will only work if keys are provided)
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("--- Supabase Client Initialized ---")
else:
    print("WARNING: Supabase credentials not found. /sync-predictions will not work.")

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Definisikan nama fitur agar sesuai saat training
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
        print(f"--- Model and Scaler loaded! Support Option B with {API_KEY_NAME} ---")
    except Exception as e:
        print(f"Error loading assets: {e}")
        raise RuntimeError("Failed to load model assets")
    yield

# --- 3. Inisialisasi FastAPI ---
app = FastAPI(
    title="Churn Prediction API - Supabase",
    lifespan=lifespan
)

# --- 4. Middleware CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        detail="Akses Ditolak: API Key tidak valid"
    )

# --- 6. Pydantic Models ---
class PredictionRequest(BaseModel):
    user_id: str
    login_freq: int
    last_login_days: int
    total_transactions: int
    avg_session_time: float

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
    return {"message": "Option B: Churn Prediction API with Supabase integration is running!"}

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(get_api_key)])
async def predict(data: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_data = pd.DataFrame([[
            data.login_freq, 
            data.last_login_days, 
            data.total_transactions, 
            data.avg_session_time
        ]], columns=FEATURE_NAMES)
        
        scaled_features = scaler.transform(input_data)
        prob = model.predict_proba(scaled_features)[0][1]
        
        return PredictionResponse(
            user_id=data.user_id,
            churn_probability=round(float(prob), 4),
            risk_level=get_risk_level(prob)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync-predictions", dependencies=[Depends(get_api_key)])
async def sync_predictions():
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized. Check your credentials.")
    
    try:
        # 1. Ambil data dari Supabase yang hasil prediksinya masih kosong
        response = supabase.table("users_churn_data").select("*").is_("churn_probability", "NULL").execute()
        users = response.data

        if not users:
            return {"message": "All user data is already up to date.", "synced_count": 0}

        results = []
        for user in users:
            # 2. Siapkan data untuk prediksi
            input_df = pd.DataFrame([[
                user['login_freq'],
                user['last_login_days'],
                user['total_transactions'],
                user['avg_session_time']
            ]], columns=FEATURE_NAMES)

            # 3. Prediksi
            scaled = scaler.transform(input_df)
            prob = float(model.predict_proba(scaled)[0][1])
            risk = get_risk_level(prob)

            # 4. Update kembali ke Supabase berdasarkan ID record
            supabase.table("users_churn_data").update({
                "churn_probability": round(prob, 4),
                "risk_level": risk
            }).eq("id", user['id']).execute()

            results.append({"user_id": user.get('username', user['id']), "risk": risk})

        return {
            "message": f"Successfully synced {len(results)} users.",
            "synced_count": len(results),
            "details": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

