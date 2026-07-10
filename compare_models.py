import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score
import numpy as np

# Import berbagai Algoritma Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Import fungsi preprocessing dari program utama agar tidak mengulang kode
import main_golden_cross as mgc

def run_comparison(df):
    print("\n" + "=" * 60)
    print("🔬 MEMULAI KOMPARASI ALGORITMA MACHINE LEARNING")
    print("=" * 60)
    
    # =========================================================================
    # LANGKAH 1: SELEKSI FITUR (FEATURE SELECTION)
    # =========================================================================
    # Menggunakan 12 fitur (konsisten dengan main_golden_cross.py untuk fair-comparison)
    features = [
        'Return_1d', 'Return_5d', 'RSI', 'MACD', 'BB_Pos', 'volume',
        'ma_gap_pct', 'ma_slope', 'Stochastic', 'ATR', 'MA_20', 'MA_100'
    ]
    df_clean = df.dropna(subset=features + ['Target_GC_Tomorrow', 'MA_Long'])
    
    X = df_clean[features].values
    y = df_clean['Target_GC_Tomorrow'].values
    
    if sum(y) < 5:
        print("Gagal: Kejadian Golden Cross terlalu sedikit.")
        return
        
    # =========================================================================
    # LANGKAH 2: PARTISI DATA (DATA SPLITTING)
    # =========================================================================
    # Split standar industri: 70% Train, 15% Val, 15% Test
    X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # =========================================================================
    # LANGKAH 3: NORMALISASI DAN PENANGANAN KETIDAKSEIMBANGAN KELAS
    # =========================================================================
    # Standardisasi fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_temp)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Undersampling Data Train (Rasio 1:2)
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train_scaled, y_train_temp)
    
    # =========================================================================
    # LANGKAH 4: GENERASI DATASET UNTUK SKENARIO PENGUJIAN LANJUTAN
    # =========================================================================
    # Skenario 2: Data Out-of-Time (Data masa depan berdasarkan kronologi)
    df_oot = df_clean.groupby('kode', group_keys=False).apply(lambda x: x.tail(max(1, int(len(x)*0.15))))
    X_oot = df_oot[features].values
    y_oot = df_oot['Target_GC_Tomorrow'].values
    X_oot_scaled = scaler.transform(X_oot)
    
    # Skenario 3: Data Noise pada Test Set (Gaussian Noise 5%)
    std_devs = np.std(X_test, axis=0)
    noise = np.random.normal(0, std_devs * 0.05, X_test.shape)
    X_test_noisy = X_test + noise
    X_test_noisy_scaled = scaler.transform(X_test_noisy)

    # =========================================================================
    # LANGKAH 5: INISIALISASI KANDIDAT ALGORITMA
    # =========================================================================
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    }

    # =========================================================================
    # LANGKAH 6: PELATIHAN DAN BENCHMARKING (EVALUASI)
    # =========================================================================
    print(f"Mengevaluasi {len(models)} model menggunakan 12 fitur indikator teknikal (konsisten dengan model utama).\n")
    
    def print_metrics(set_name, y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred) * 100
        auc = roc_auc_score(y_true, y_prob) if sum(y_true) > 0 else 0
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"   {set_name:<20} | Acc: {acc:>6.2f}% | AUC: {auc:.3f} | Prec(1): {prec:.3f} | Rec(1): {rec:.3f} | F1(1): {f1:.3f}")

    for name, model in models.items():
        print("-" * 85)
        print(f"Model: {name}")
        model.fit(X_train_res, y_train_res)
        
        # Validation
        y_val_pred = model.predict(X_val_scaled)
        if hasattr(model, "predict_proba"):
            y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
        else:
            y_val_prob = y_val_pred # Untuk model tanpa probability, ini hanya fallback saja. SVM disini sdh pake probability=True
            
        print_metrics("Validation", y_val, y_val_pred, y_val_prob)
        
        # Skenario 1: Test Set Standar
        y_test_pred = model.predict(X_test_scaled)
        y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
        print_metrics("Skenario 1 (Test)", y_test, y_test_pred, y_test_prob)
        
        # Skenario 2: Out of Time
        y_oot_pred = model.predict(X_oot_scaled)
        y_oot_prob = model.predict_proba(X_oot_scaled)[:, 1]
        print_metrics("Skenario 2 (OOT)", y_oot, y_oot_pred, y_oot_prob)
        
        # Skenario 3: Test dengan Noise
        y_noisy_pred = model.predict(X_test_noisy_scaled)
        y_noisy_prob = model.predict_proba(X_test_noisy_scaled)[:, 1]
        print_metrics("Skenario 3 (Noise)", y_test, y_noisy_pred, y_noisy_prob)
        
    print("-" * 85)
    print("\nKESIMPULAN KOMPARASI METRIK KELAS POSITIF (GOLDEN CROSS):")
    print("Perhatikan konsistensi antar skenario pengujian:")
    print("- Validation vs Test Standar: Mengukur terjadinya overfit.")
    print("- Out-Of-Time: Mengukur apakah performa bertahan pada data 'masa depan' tulen.")
    print("- Noise: Mengukur kestabilan model terhadap fluktuasi/gangguan kecil di harga.")
    print("\nMetrik Fokus Kelas Positif (Golden Cross):")
    print("- Precision mengukur akurasi tebakan (mencegah False Positive / jebakan beli).")
    print("- Recall mengukur seberapa banyak Golden Cross asli yang ditangkap (mencegah False Negative).")

# ==============================================================================
# PIPELINE EKSEKUSI KOMPARASI (LINEAR EXECUTION)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modul Komparasi Algoritma Machine Learning")
    parser.add_argument('--file', type=str, default='data_tester/daridosenbarukhususgoldencross.csv', help='Nama file CSV.')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"❌ File '{args.file}' tidak ditemukan.")
    else:
        # ----------------------------------------------------------------------
        # PEMANGGILAN FUNGSI SECARA BERURUTAN (LINEAR) DARI MODUL UTAMA
        # ----------------------------------------------------------------------
        # LANGKAH 1: Ingestion
        df_raw = mgc.load_data(args.file)
        
        # LANGKAH 2: Ekstraksi Fitur
        df_feat = mgc.preprocess_and_extract_features(df_raw)
        
        # LANGKAH 3: Definisi Label Target
        df_target = mgc.define_target(df_feat)
        
        # LANGKAH 4: Menjalankan Loop Komparasi dan Evaluasi Benchmark
        run_comparison(df_target)
