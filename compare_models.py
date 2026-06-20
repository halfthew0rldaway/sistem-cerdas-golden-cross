import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, roc_auc_score

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
    
    # 1. Pilih Fitur
    features = ['Return_1d', 'Return_5d', 'RSI', 'MACD', 'BB_Pos', 'volume']
    df_clean = df.dropna(subset=features + ['Target_GC_Tomorrow', 'MA_Long'])
    
    X = df_clean[features].values
    y = df_clean['Target_GC_Tomorrow'].values
    
    if sum(y) < 5:
        print("Gagal: Kejadian Golden Cross terlalu sedikit.")
        return
        
    # 2. Split & Standardisasi & Undersampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train_scaled, y_train)

    # 3. Daftar Model yang akan ditandingkan
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "Support Vector Machine (SVM)": SVC(class_weight='balanced', probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42)
    }

    # 4. Latih dan Evaluasi setiap model
    results = []
    print(f"{'Algoritma':<30} | {'Akurasi':<10} | {'ROC-AUC':<10}")
    print("-" * 60)
    
    for name, model in models.items():
        # Latih model
        model.fit(X_train_res, y_train_res)
        
        # Ujian
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Hitung metrik
        acc = accuracy_score(y_test, y_pred) * 100
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({"Algoritma": name, "Akurasi (%)": acc, "ROC-AUC": auc})
        print(f"{name:<30} | {acc:>7.2f} % | {auc:>8.3f}")
        
    print("\nKESIMPULAN:")
    print("Random Forest biasanya memiliki keseimbangan terbaik antara Akurasi dan ROC-AUC,")
    print("serta lebih tahan terhadap overfitting dibandingkan Decision Tree tunggal.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='transaksi_harian_202606130928(daridosenbarukhususgoldencross).csv', help='Nama file CSV.')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File '{args.file}' tidak ditemukan.")
    else:
        # Load dan Preprocess meminjam fungsi dari main_golden_cross.py
        df_raw = mgc.load_data(args.file)
        df_feat = mgc.preprocess_and_extract_features(df_raw)
        df_target = mgc.define_target(df_feat)
        
        # Jalankan komparasi
        run_comparison(df_target)
