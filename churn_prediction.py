import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Set seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=1000):
    """
    Menghasilkan dataset sintetis untuk prediksi churn dengan informasi identitas.
    Logika: User yang jarang login, lama tidak login, dan transaksi sedikit lebih cenderung churn.
    """
    print("--- 1. Generating Synthetic Dataset ---")
    
    # Fitur-fitur identitas
    usernames = [f"user_{i}" for i in range(1, n_samples + 1)]
    emails = [f"user_{i}@example.com" for i in range(1, n_samples + 1)]
    no_wa = [f"62812{np.random.randint(10000000, 99999999)}" for _ in range(n_samples)]
    
    # Fitur perilaku
    login_freq = np.random.randint(1, 31, n_samples)
    last_login_days = np.random.randint(0, 60, n_samples)
    total_transactions = np.random.randint(0, 100, n_samples)
    avg_session_time = np.random.uniform(5, 60, n_samples)
    
    # Logika Churn (Simulasi)
    logit = (
        0.5 * (30 - login_freq) +
        0.3 * last_login_days -
        0.1 * total_transactions -
        0.05 * avg_session_time -
        10
    )
    
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (prob_churn > 0.5).astype(int)
    
    df = pd.DataFrame({
        'username': usernames,
        'email': emails,
        'no_wa': no_wa,
        'login_freq': login_freq,
        'last_login_days': last_login_days,
        'total_transactions': total_transactions,
        'avg_session_time': avg_session_time,
        'churn': churn
    })
    
    df.to_csv('user_churn_data.csv', index=False)
    print(f"Dataset berhasil dibuat dengan {n_samples} baris: 'user_churn_data.csv'\n")
    return df

def preprocess_and_train(df):
    print("--- 2. Preprocessing Data ---")
    # Drop kolom identitas karena tidak digunakan untuk prediksi
    X = df.drop(['churn', 'username', 'email', 'no_wa'], axis=1)
    y = df['churn']
    
    # Split data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data Split: Train {len(X_train)}, Test {len(X_test)}")
    
    # Scaling (Sangat penting untuk Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Fitur berhasil di-scaling menggunakan StandardScaler.\n")
    
    print("--- 3. Training Logistic Regression ---")
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    print("Model berhasil dilatih.\n")
    
    return model, scaler, X_test_scaled, y_test, X.columns

def evaluate_model(model, X_test_scaled, y_test):
    print("--- 4. Evaluasi Model ---")
    y_pred = model.predict(X_test_scaled)
    
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Confusion Matrix
    print("--- 5. Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Aktif', 'Churn'], yticklabels=['Aktif', 'Churn'])
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix disimpan sebagai 'confusion_matrix.png'\n")

def show_feature_importance(model, feature_names):
    print("--- 6. Feature Importance (Coefficients) ---")
    # Logistic Regression menggunakan koefisien sebagai indikator pengaruh fitur
    importance = model.coef_[0]
    feat_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    
    print(feat_importance)
    
    plt.figure(figsize=(8, 5))
    feat_importance.plot(kind='barh', color='skyblue')
    plt.title('Feature Importance (Logistic Regression Coefficients)')
    plt.xlabel('Coefficient Value')
    plt.savefig('feature_importance.png')
    print("Visualisasi Feature Importance disimpan sebagai 'feature_importance.png'\n")

def save_model(model, scaler):
    print("--- 7. Menyimpan Model ---")
    joblib.dump(model, 'churn_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model ('churn_model.joblib') dan Scaler ('scaler.joblib') berhasil disimpan.\n")

def predict_new_user():
    print("--- 8. Prediksi Data User Baru ---")
    # Load model and scaler
    model = joblib.load('churn_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # Contoh data 1 user baru
    # Fitur: [login_freq, last_login_days, total_transactions, avg_session_time]
    new_user_data = np.array([[2, 45, 1, 10.5]])  # User yang jarang login dan sudah lama tidak aktif
    
    # Scaling data baru sebelum prediksi
    new_user_scaled = scaler.transform(new_user_data)
    
    # Prediksi
    prediction = model.predict(new_user_scaled)
    prob = model.predict_proba(new_user_scaled)[:, 1]
    
    status = "CHURN" if prediction[0] == 1 else "AKTIF"
    print(f"Data Input: {new_user_data}")
    print(f"Hasil Prediksi: {status}")
    print(f"Probabilitas Churn: {prob[0]:.4f}")

if __name__ == "__main__":
    # Jalankan semua tahap
    data = generate_synthetic_data(1000)
    trained_model, data_scaler, X_test, y_test, features = preprocess_and_train(data)
    evaluate_model(trained_model, X_test, y_test)
    show_feature_importance(trained_model, features)
    save_model(trained_model, data_scaler)
    predict_new_user()
