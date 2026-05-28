"""
Langkah 5: Model Prediksi
Membuat model machine learning untuk memprediksi Golden Cross.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


def build_model(df, model_type='random_forest', test_size=0.2):
    """
    Buat model prediksi Golden Cross.
    
    Model memprediksi apakah akan terjadi Golden Cross berdasarkan
    fitur-fitur teknikal (MA50, MA200, selisih MA, harga Close).
    
    Parameters:
        df (pd.DataFrame): DataFrame dengan fitur dan sinyal
        model_type (str): 'random_forest' atau 'logistic_regression'
        test_size (float): Proporsi data testing (default: 0.2)
        
    Returns:
        tuple: (model, accuracy, report, scaler)
    """
    print("=" * 50)
    print("🤖 LANGKAH 5: Model Prediksi")
    print("=" * 50)
    
    # Siapkan fitur dan target
    features = ['MA50', 'MA200', 'MA_Diff', 'Close']
    
    # Hapus baris dengan NaN (MA belum terhitung)
    df_clean = df.dropna(subset=features + ['Signal'])
    
    print(f"\n📊 Data untuk model: {len(df_clean)} baris")
    print(f"   Fitur (X): {features}")
    print(f"   Target (y): Signal (1=Golden Cross, 0=Tidak)")
    
    X = df_clean[features].values
    y = df_clean['Signal'].values
    
    # Info distribusi kelas
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    print(f"\n📈 Distribusi target:")
    print(f"   Golden Cross (1): {n_positive} ({n_positive/len(y)*100:.2f}%)")
    print(f"   Tidak ada (0): {n_negative} ({n_negative/len(y)*100:.2f}%)")
    
    # Bagi data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if n_positive >= 2 else None
    )
    
    print(f"\n📦 Pembagian data:")
    print(f"   Training: {len(X_train)} baris ({(1-test_size)*100:.0f}%)")
    print(f"   Testing:  {len(X_test)} baris ({test_size*100:.0f}%)")
    
    # Normalisasi fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Pilih dan latih model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        model_name = "Random Forest"
    else:
        model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        model_name = "Logistic Regression"
    
    print(f"\n🔧 Model: {model_name}")
    print(f"   Training model...")
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print(f"\n📊 HASIL EVALUASI:")
    print(f"   Akurasi: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n   Classification Report:")
    print(f"   {'-'*45}")
    for line in report.split('\n'):
        print(f"   {line}")
    
    # Feature importance (untuk Random Forest)
    if model_type == 'random_forest':
        importances = model.feature_importances_
        print(f"\n   Feature Importance:")
        for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
            bar = '█' * int(imp * 30)
            print(f"   {feat:10s}: {imp:.4f} {bar}")
    
    print()
    return model, accuracy, report, scaler


if __name__ == "__main__":
    from load_data import load_data
    from preprocess import preprocess
    from features import add_moving_averages
    from detect import detect_golden_cross
    
    df = load_data()
    df = preprocess(df)
    df = add_moving_averages(df)
    df = detect_golden_cross(df)
    model, acc, report, scaler = build_model(df)
