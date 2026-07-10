"""
PIPELINE LENGKAP DETEKSI GOLDEN CROSS - BEST PRACTICE ML
=============================================================================
File ini adalah program utama untuk mendeteksi pola "Golden Cross" pada pergerakan saham.
Golden Cross adalah pola di mana rata-rata pergerakan jangka pendek (MA20) 
memotong rata-rata pergerakan jangka panjang (MA50) ke arah atas.

Kode ini dirancang agar mudah dibaca oleh pemula, dengan komentar bahasa Indonesia
di setiap langkah penting. Program ini menggunakan Machine Learning (Random Forest)
untuk MEMPREDIKSI apakah Golden Cross akan terjadi besok berdasarkan indikator hari ini.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import argparse

# Pustaka untuk visualisasi grafik interaktif
import plotly.graph_objects as go

# Pustaka untuk Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Pustaka untuk mengatasi data tidak seimbang (Imbalanced Data)
# RandomUnderSampler membuang data mayoritas secara acak agar jumlahnya seimbang
from imblearn.under_sampling import RandomUnderSampler

# ==========================================
# LANGKAH 1: MEMUAT DATA
# ==========================================
def load_data(filepath):
    """Fungsi untuk membaca data CSV ke dalam format DataFrame Pandas."""
    print("=" * 60)
    print(" LANGKAH 1: Membaca Data dari CSV")
    print("=" * 60)
    
    # Membaca file CSV (Perhatikan: separatornya adalah titik koma ';')
    df = pd.read_csv(filepath, sep=';')
    print(f" Data berhasil dimuat: {len(df)} baris")
    
    return df

# ==========================================
# LANGKAH 2 & 3: PREPROCESSING & FITUR INDIKATOR
# ==========================================
def preprocess_and_extract_features(df, ma_short=50, ma_long=200):
    """
    Membersihkan data dan membuat Indikator Teknikal.
    Indikator Teknikal adalah rumus matematika yang digunakan trader untuk analisis.
    """
    print("\n" + "=" * 60)
    print(" LANGKAH 2 & 3: Preprocessing & Ekstraksi Fitur (Indikator)")
    print("=" * 60)
    
    # Copy data agar tidak merusak data aslinya
    df = df.copy()
    
    # 1. Pastikan kolom tanggal berformat 'datetime' lalu urutkan
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df = df.sort_values(['kode', 'tanggal'])
    df = df.dropna(subset=['close_price'])
    
    # 2. MENGHITUNG MOVING AVERAGE (Dinamis sesuai Input)
    # Default: MA50 dan MA200 (Golden Cross Sejati). 
    # Jika data kurang, user bisa menyesuaikan argumen saat menjalankan program.
    df['MA_Short'] = df.groupby('kode')['close_price'].transform(lambda x: x.rolling(window=ma_short).mean())
    df['MA_Long'] = df.groupby('kode')['close_price'].transform(lambda x: x.rolling(window=ma_long).mean())
    
    # 3. MENGHITUNG INDIKATOR TAMBAHAN UNTUK MACHINE LEARNING
    # Model ML butuh indikator murni (RSI, MACD) agar tidak sekadar 'menjiplak' rumus MA.
    
    # A. Return (Persentase kenaikan/penurunan harga harian dan mingguan)
    df['Return_1d'] = df.groupby('kode')['close_price'].pct_change()
    df['Return_5d'] = df.groupby('kode')['close_price'].pct_change(5)
    
    # B. RSI (Relative Strength Index) - Indikator Jenuh Beli / Jenuh Jual
    def calc_rsi(x, w=14):
        delta = x.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    df['RSI'] = df.groupby('kode')['close_price'].transform(calc_rsi)
    
    # C. MACD (Moving Average Convergence Divergence) - Indikator Momentum
    def calc_macd(x):
        return x.ewm(span=12).mean() - x.ewm(span=26).mean()
    df['MACD'] = df.groupby('kode')['close_price'].transform(calc_macd)
    
    # D. Bollinger Bands (Indikator Volatilitas)
    df['BB_Mid'] = df.groupby('kode')['close_price'].transform(lambda x: x.rolling(20).mean())
    df['BB_Std'] = df.groupby('kode')['close_price'].transform(lambda x: x.rolling(20).std())
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    
    # Menghitung posisi harga relatif terhadap lebar Bollinger Bands
    df['BB_Pos'] = (df['close_price'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # E. FITUR BARU 1: Derivative Moving Average (Menangkap Momentum tanpa Data Leakage)
    # Menghitung persentase jarak antara MA Pendek dan MA Panjang untuk melihat seberapa dekat persilangan
    df['ma_gap_pct'] = (df['MA_Short'] - df['MA_Long']) / df['MA_Long'] * 100
    # Menghitung kecepatan/kemiringan tren MA Pendek (slope) dalam 5 hari terakhir
    df['ma_slope'] = df['MA_Short'] - df.groupby('kode')['MA_Short'].shift(5)

    # F. FITUR BARU 2: Indikator Tambahan (Stochastic Oscillator & ATR)
    # Stochastic Oscillator: Mengukur apakah harga sudah jenuh beli/jual secara absolut dibanding titik tertinggi-terendah 14 hari
    low_min = df.groupby('kode')['low_price'].transform(lambda x: x.rolling(window=14).min())
    high_max = df.groupby('kode')['high_price'].transform(lambda x: x.rolling(window=14).max())
    df['Stochastic'] = 100 * (df['close_price'] - low_min) / (high_max - low_min)
    
    # ATR (Average True Range): Mengukur tingkat volatilitas (keliaran fluktuasi harga) pasar selama 14 hari
    tr1 = abs(df['high_price'] - df['low_price'])
    tr2 = abs(df['high_price'] - df.groupby('kode')['close_price'].shift(1))
    tr3 = abs(df['low_price'] - df.groupby('kode')['close_price'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = true_range.groupby(df['kode']).transform(lambda x: x.rolling(window=14).mean())

    # G. FITUR BARU 3: Moving Average Periode Lain (Konteks Tren Tambahan)
    # Menambah MA20 (tren sangat pendek) dan MA100 (tren menengah) sebagai fitur ekstra
    df['MA_20'] = df.groupby('kode')['close_price'].transform(lambda x: x.rolling(window=20).mean())
    df['MA_100'] = df.groupby('kode')['close_price'].transform(lambda x: x.rolling(window=100).mean())
    
    print(f" Semua indikator (MA{ma_short}, MA{ma_long}, RSI, MACD, BB, Stochastic, ATR, dll) telah dihitung!")
    return df

# ==========================================
# LANGKAH 4: MENYIAPKAN TARGET PREDIKSI (Y)
# ==========================================
def define_target(df, ma_short=50, ma_long=200):
    """
    Fungsi ini mencari kapan Golden Cross terjadi, HANYA untuk dijadikan label.
    Penting: ML harus memprediksi MASA DEPAN, bukan HARI INI.
    """
    print("\n" + "=" * 60)
    print(" LANGKAH 4: Menentukan Target (Label) Machine Learning")
    print("=" * 60)
    
    # 1. Cari titik perpotongan (Golden Cross) yang terjadi HARI INI
    # Syarat: MA_Short lebih besar dari MA_Long hari ini, TAPI kemarin MA_Short masih di bawah MA_Long
    cond1 = df['MA_Short'] > df['MA_Long']
    cond2 = df.groupby('kode')['MA_Short'].shift(1) <= df.groupby('kode')['MA_Long'].shift(1)
    df['Is_GC_Today'] = (cond1 & cond2).astype(int)
    
    # 2. MENCEGAH DATA LEAKAGE (BOCORAN KUNCI JAWABAN)
    # Jika kita menebak Golden Cross HARI INI menggunakan data HARI INI, itu namanya "Data Leakage"
    # dan akurasinya pasti tidak masuk akal (100% palsu).
    # SOLUSI: Kita minta ML menebak kejadian BESOK (H+1) berdasarkan data HARI INI.
    df['Target_GC_Tomorrow'] = df.groupby('kode')['Is_GC_Today'].shift(-1).fillna(0)
    
    total_gc = df['Is_GC_Today'].sum()
    print(f" Total Golden Cross (MA{ma_short} memotong MA{ma_long}) terdeteksi di seluruh data: {total_gc} kali.")
    print(" Target AI digeser (di-shift) 1 hari ke depan agar AI murni melakukan Forecasting (Meramal Masa Depan).")
    
    return df

# ==========================================
# LANGKAH 5: MELATIH MACHINE LEARNING
# ==========================================

def print_eval_metrics(y_true, y_pred, y_prob, title):
    """Helper function untuk mencetak metrik evaluasi secara rapih (Best Practice: DRY).
    Mengembalikan dict berisi semua metrik untuk disimpan ke JSON."""
    acc = accuracy_score(y_true, y_pred) * 100
    auc = roc_auc_score(y_true, y_prob) if sum(y_true) > 0 else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Hitung Precision & Recall kelas 1 secara manual
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tn = int(cm[0, 0])
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    support_pos = int(sum(y_true))
    support_neg = int(len(y_true) - sum(y_true))
    
    print(f"\n --- {title} ---")
    print(f"   Accuracy : {acc:.2f}%")
    if sum(y_true) > 0:
        print(f"   ROC-AUC  : {auc:.3f}")
    else:
        print("   ROC-AUC  : N/A (Tidak ada target 1 di data ini)")
    
    print(f"\n   [Classification Report - {title}]")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print(f"   [Confusion Matrix - {title}]")
    print(f"   True Negative  (TN) : {tn:<5} | False Positive (FP) : {fp}")
    print(f"   False Negative (FN) : {fn:<5} | True Positive  (TP) : {tp}\n")
    
    return {
        "accuracy": round(acc, 2),
        "roc_auc": round(auc, 3),
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "support_pos": support_pos, "support_neg": support_neg
    }

def train_model(df):
    """Melatih algoritma Random Forest untuk mencari pola di data teknikal."""
    print("\n" + "=" * 60)
    print(" LANGKAH 5: Melatih Machine Learning (Random Forest)")
    print("=" * 60)
    
    # 1. Pilih Fitur (X) yang akan dipelajari oleh model
    # Memasukkan semua 12 fitur (kolom) baru untuk memperkaya kemampuan belajar AI
    features = [
        'Return_1d', 'Return_5d', 'RSI', 'MACD', 'BB_Pos', 'volume',
        'ma_gap_pct', 'ma_slope', 'Stochastic', 'ATR', 'MA_20', 'MA_100'
    ]
    
    # Buang baris data yang ada nilai kosong (NaN)
    df_clean = df.dropna(subset=features + ['Target_GC_Tomorrow', 'MA_Long'])
    X = df_clean[features].values
    y = df_clean['Target_GC_Tomorrow'].values
    
    if sum(y) < 5:
        print(" Gagal: Kejadian Golden Cross terlalu sedikit untuk melatih ML.")
        return None, None
        
    # 2. Bagi data: 70% Train, 15% Validation, 15% Testing
    # Pertama pisahkan 30% untuk Validation & Test secara Stratified
    X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    # Kemudian pisahkan sisa 30% menjadi 15% Val dan 15% Test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 3. Standardisasi: Menyamakan skala angka (misal, harga puluhan ribu vs persentase desimal)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_temp)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. MENGATASI IMBALANCED DATA (PENTING!)
    print(" Menerapkan Undersampling: Membuang sebagian data kelas 0 agar seimbang dengan kelas 1...")
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train_scaled, y_train_temp)
    
    # 5. Latih Model Random Forest
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    
    # 6. Evaluasi 4 Skenario Pengujian
    all_metrics = {}
    total_rows = len(df)
    total_gc = int(df['Is_GC_Today'].sum())
    
    # Validation Set
    y_val_pred = model.predict(X_val_scaled)
    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
    all_metrics['validation'] = print_eval_metrics(y_val, y_val_pred, y_val_prob, "EVALUASI VALIDATION SET (15%)")

    # Test Set Standar
    y_test_pred = model.predict(X_test_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
    all_metrics['test_standar'] = print_eval_metrics(y_test, y_test_pred, y_test_prob, "SKENARIO 1: EVALUASI PADA TEST SET STANDAR (15%)")

    # Out-Of-Time (15% Terbaru Kronologis)
    # Mengambil 15% data terbaru dari setiap kode saham tanpa mengacak
    df_oot = df_clean.groupby('kode', group_keys=False).apply(lambda x: x.tail(max(1, int(len(x)*0.15))))
    X_oot = df_oot[features].values
    y_oot = df_oot['Target_GC_Tomorrow'].values
    
    X_oot_scaled = scaler.transform(X_oot)
    y_oot_pred = model.predict(X_oot_scaled)
    y_oot_prob = model.predict_proba(X_oot_scaled)[:, 1]
    all_metrics['oot'] = print_eval_metrics(y_oot, y_oot_pred, y_oot_prob, "SKENARIO 2: EVALUASI OUT-OF-TIME (15% TERBARU KRONOLOGIS)")

    # Noise Pada Test Set
    # Tambahkan noise sebesar 5% dari standar deviasi masing-masing fitur ke test set
    std_devs = np.std(X_test, axis=0)
    noise = np.random.normal(0, std_devs * 0.05, X_test.shape)
    X_test_noisy = X_test + noise
    X_test_noisy_scaled = scaler.transform(X_test_noisy)
    
    y_noisy_pred = model.predict(X_test_noisy_scaled)
    y_noisy_prob = model.predict_proba(X_test_noisy_scaled)[:, 1]
    all_metrics['noise'] = print_eval_metrics(y_test, y_noisy_pred, y_noisy_prob, "SKENARIO 3: EVALUASI DENGAN NOISE PADA TEST SET")
    
    # Simpan semua metrik ke JSON agar bisa dibaca oleh Web Dashboard
    json_output = {
        "dataset": {"total_rows": total_rows, "total_gc": total_gc},
        "scenarios": all_metrics
    }
    with open('metrics_results.json', 'w') as f:
        json.dump(json_output, f, indent=2)
    print(" Metrik evaluasi telah disimpan ke metrics_results.json")
    
    return model, scaler

# ==========================================
# LANGKAH 6: VISUALISASI INTERAKTIF (HTML)
# ==========================================
def plot_chart(df_all, target_saham, save_path='golden_cross_interactive.html', ma_short=50, ma_long=200):
    """Membuat grafik Candlestick interaktif yang bisa di-zoom/di-geser lewat Browser."""
    print("\n" + "=" * 60)
    print(f" LANGKAH 6: Visualisasi Chart Saham '{target_saham.upper()}'")
    print("=" * 60)
    
    # Ambil data spesifik 1 saham saja untuk di-plot
    df = df_all[df_all['kode'].str.lower() == target_saham.lower()].copy()
    if len(df) == 0:
        print(f" Saham {target_saham.upper()} tidak ditemukan di data CSV.")
        return
        
    df = df.set_index('tanggal')
    fig = go.Figure()

    # A. Gambar Candlestick (Batang Harga)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open_price'], high=df['high_price'],
        low=df['low_price'], close=df['close_price'], name='Candlestick Harga',
        increasing_line_color='#00ff88', decreasing_line_color='#ff6b6b'
    ))

    # B. Gambar Garis Moving Average Dinamis
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_Short'], line=dict(color='#00d2ff', width=2), name=f'MA{ma_short} (Garis Cepat)'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_Long'], line=dict(color='#ffaa00', width=2), name=f'MA{ma_long} (Garis Lambat)'))

    # C. Beri Tanda Segitiga Khusus pada Hari di mana Terjadi Golden Cross
    gc = df[df['Is_GC_Today'] == 1]
    if len(gc) > 0:
        fig.add_trace(go.Scatter(
            x=gc.index, y=gc['close_price'], mode='markers+text',
            marker=dict(symbol='triangle-up', color='magenta', size=16),
            text=["GC"] * len(gc), textposition="top center",
            textfont=dict(color="magenta", size=12, family="Arial Black"), name='Golden Cross Point'
        ))
        
        # Cetak tanggal kejadian ke terminal
        for date in gc.index:
            print(f"    Golden Cross terdeteksi di grafik pada tanggal: {date.strftime('%d %B %Y')}")

    # Percantik tampilan layar
    fig.update_layout(
        title=f' Analisis Deteksi Golden Cross Saham {target_saham.upper()}',
        yaxis_title='Harga Rupiah (IDR)', xaxis_title='Tanggal Transaksi',
        template='plotly_dark', xaxis_rangeslider_visible=True,
        hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Simpan ke komputer
    fig.write_html(save_path)
    print(f"\n Selesai! Grafik visualisasi interaktif telah disimpan di: '{save_path}'")
    print(f" Buka file tersebut menggunakan Browser (Chrome/Edge/Safari) untuk melihat hasilnya.")

# ==============================================================================
# PIPELINE EKSEKUSI UTAMA (LINEAR EXECUTION)
# ==============================================================================
# Bagian ini adalah titik masuk saat program dijalankan lewat terminal, dirancang
# secara terstruktur dan linear dari ekstraksi data hingga visualisasi hasil.
if __name__ == "__main__":
    # Inisialisasi argumen CLI agar parameter dinamis tanpa mengubah source code
    parser = argparse.ArgumentParser(description="Pipeline Machine Learning Prediksi Golden Cross")
    parser.add_argument('--file', type=str, default='data_tester/daridosenbarukhususgoldencross.csv', help='Nama file CSV masukan.')
    parser.add_argument('--saham', type=str, default='bmtr', help='Kode Saham untuk dirender grafiknya (contoh: adro, bmtr)')
    parser.add_argument('--ma-short', type=int, default=50, help='Periode Moving Average Cepat (Default: 50)')
    parser.add_argument('--ma-long', type=int, default=200, help='Periode Moving Average Lambat (Default: 200)')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"❌ File '{args.file}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    else:
        # ----------------------------------------------------------------------
        # PIPELINE MACHINE LEARNING - DIEKSEKUSI SECARA BERURUTAN (LINEAR)
        # ----------------------------------------------------------------------
        
        # LANGKAH 1: Data Ingestion (Membaca data mentah dari sumber CSV)
        df_raw = load_data(args.file)
        
        # LANGKAH 2 & 3: Feature Engineering (Mengekstrak 12 indikator teknikal)
        df_feat = preprocess_and_extract_features(df_raw, ma_short=args.ma_short, ma_long=args.ma_long)
        
        # LANGKAH 4: Target Definition (Menandai dan menggeser label T+1)
        df_target = define_target(df_feat, ma_short=args.ma_short, ma_long=args.ma_long)
        
        # LANGKAH 5: Model Training & Evaluation (Pelatihan, Undersampling, & Metrik)
        model, scaler = train_model(df_target)
        
        # LANGKAH 6: Data Visualization (Merender grafik interaktif berbasis HTML)
        plot_chart(df_target, args.saham, ma_short=args.ma_short, ma_long=args.ma_long)