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
    
    print(f" Semua indikator (MA{ma_short}, MA{ma_long}, RSI, MACD, Bollinger Bands) telah dihitung!")
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
def train_model(df):
    """Melatih algoritma Random Forest untuk mencari pola di data teknikal."""
    print("\n" + "=" * 60)
    print(" LANGKAH 5: Melatih Machine Learning (Random Forest)")
    print("=" * 60)
    
    # 1. Pilih Fitur (X) yang akan dipelajari oleh model
    # (Kita sengaja tidak memasukkan MA agar AI berpikir mandiri mencari pola)
    features = ['Return_1d', 'Return_5d', 'RSI', 'MACD', 'BB_Pos', 'volume']
    
    # Buang baris data yang ada nilai kosong (NaN)
    df_clean = df.dropna(subset=features + ['Target_GC_Tomorrow', 'MA_Long'])
    X = df_clean[features].values
    y = df_clean['Target_GC_Tomorrow'].values
    
    if sum(y) < 5:
        print("⚠ Gagal: Kejadian Golden Cross terlalu sedikit untuk melatih ML.")
        return None, None
        
    # 2. Bagi data: 80% untuk Latihan (Train), 20% untuk Ujian (Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Standardisasi: Menyamakan skala angka (misal, harga puluhan ribu vs persentase desimal)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. MENGATASI IMBALANCED DATA (PENTING!)
    # Data '0' (Tidak ada GC) ada ratusan ribu. Data '1' (Ada GC) cuma ratusan.
    # Jika tidak diimbangi, model hanya akan menebak '0' terus menerus.
    print("⚙ Menerapkan Undersampling: Membuang sebagian data kelas 0 agar seimbang dengan kelas 1...")
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train_scaled, y_train)
    
    # 5. Latih Model Random Forest
    # class_weight='balanced' memberikan porsi perhatian lebih besar pada pola yang langka
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    
    # 6. Evaluasi Model (Ujian)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n --- HASIL EVALUASI REALISTIS (BUKAN AKURASI PALSU) ---")
    # Akurasi bisa menipu, karenanya kita melihat ROC-AUC. Di atas 0.7 artinya model sudah bagus.
    print(f"   Accuracy (Akurasi Global)  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"   ROC-AUC Score              : {roc_auc_score(y_test, y_prob):.3f} (Lebih tinggi = lebih baik dalam membedakan pola)")
    
    print("\n   [Confusion Matrix]")
    print("   Isinya: [Tebak 0 Benar, Tebak 1 Salah (False Positive)]")
    print("           [Tebak 0 Salah, Tebak 1 Benar (True Positive)]")
    print(confusion_matrix(y_test, y_pred))
    
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
        print(f"⚠ Saham {target_saham.upper()} tidak ditemukan di data CSV.")
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

# ==========================================
# EKSEKUSI PROGRAM UTAMA
# ==========================================
# Bagian ini adalah titik masuk saat program dijalankan lewat terminal
if __name__ == "__main__":
    # Menyiapkan argumen agar pengguna bisa mengatur file dan nama saham lewat terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='transaksi_harian_202606130928(daridosenbarukhususgoldencross).csv', help='Nama file CSV.')
    parser.add_argument('--saham', type=str, default='bmtr', help='Kode Saham yang ingin digambar (contoh: adro, bmtr)')
    parser.add_argument('--ma-short', type=int, default=50, help='Periode Moving Average Cepat (Default: 50)')
    parser.add_argument('--ma-long', type=int, default=200, help='Periode Moving Average Lambat (Default: 200)')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f" File '{args.file}' tidak ditemukan. Pastikan file ada di satu folder dengan program ini.")
    else:
        # Program otomatis beradaptasi dengan tipe data dan parameter (Universal)
        df_raw = load_data(args.file)
        df_feat = preprocess_and_extract_features(df_raw, ma_short=args.ma_short, ma_long=args.ma_long)
        df_target = define_target(df_feat, ma_short=args.ma_short, ma_long=args.ma_long)
        model, scaler = train_model(df_target)
        plot_chart(df_target, args.saham, ma_short=args.ma_short, ma_long=args.ma_long)