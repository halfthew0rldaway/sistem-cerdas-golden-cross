"""
PIPELINE LENGKAP DETEKSI GOLDEN CROSS (ADVANCED & INTERACTIVE)
Langkah 1 - 6 (Terintegrasi)
"""
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# ==========================================
# LANGKAH 1: Load Data
# ==========================================
def load_data(filepath='data/data_sample.csv'):
    print("=" * 50)
    print("📥 LANGKAH 1: Load Data")
    print("=" * 50)
    df = pd.read_csv(filepath, sep=';')
    print(f"✅ Data berhasil dimuat: {len(df)} baris")
    return df


# ==========================================
# LANGKAH 2: Preprocessing
# ==========================================
def preprocess(df, target_saham='adro'):
    print("\n" + "=" * 50)
    print("🧹 LANGKAH 2: Preprocessing & Filtering")
    print("=" * 50)
    df = df.copy()
    
    # Filter saham
    df_saham = df[df['kode'].str.lower() == target_saham.lower()].copy()
    print(f"✅ Filter saham '{target_saham.upper()}' diterapkan (Sisa: {len(df_saham)} baris)")
    
    # Format tanggal
    df_saham['tanggal'] = pd.to_datetime(df_saham['tanggal'])
    df_saham = df_saham.sort_values('tanggal')
    df_saham = df_saham.dropna(subset=['close_price'])
    df_saham = df_saham.set_index('tanggal')
    
    print("✅ Kolom 'tanggal' diubah ke datetime dan dijadikan index")
    return df_saham


# ==========================================
# LANGKAH 3: Feature Engineering (Advanced)
# ==========================================
def add_technical_indicators(df):
    print("\n" + "=" * 50)
    print("📊 LANGKAH 3: Technical Indicators")
    print("=" * 50)
    df = df.copy()
    
    # 1. Moving Averages
    df['MA50'] = df['close_price'].rolling(window=50).mean()
    df['MA200'] = df['close_price'].rolling(window=200).mean()
    df['MA_Diff'] = df['MA50'] - df['MA200']
    
    # 2. RSI (Relative Strength Index) - 14 Hari
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence)
    exp1 = df['close_price'].ewm(span=12, adjust=False).mean()
    exp2 = df['close_price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. Bollinger Bands (20 Hari)
    df['BB_Mid'] = df['close_price'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['close_price'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['close_price'].rolling(window=20).std()
    
    print("✅ MA50, MA200, RSI, MACD, dan Bollinger Bands berhasil dihitung!")
    return df


# ==========================================
# LANGKAH 4: Deteksi Golden Cross
# ==========================================
def detect_golden_cross(df):
    print("\n" + "=" * 50)
    print("🔍 LANGKAH 4: Deteksi Golden Cross")
    print("=" * 50)
    df = df.copy()
    df['Signal'] = 0
    
    cond1 = df['MA50'] > df['MA200']
    cond2 = df['MA50'].shift(1) <= df['MA200'].shift(1)
    df.loc[cond1 & cond2, 'Signal'] = 1
    
    golden_cross_dates = df[df['Signal'] == 1].index
    print(f"🎯 Jumlah Golden Cross terdeteksi: {len(golden_cross_dates)}")
    
    for i, date in enumerate(golden_cross_dates, 1):
        close_price = df.loc[date, 'close_price']
        print(f"   {i}. {date.strftime('%Y-%m-%d')} — Harga: IDR {close_price:,.0f}")
        
    return df


# ==========================================
# LANGKAH 5: Model Prediksi Machine Learning
# ==========================================
def build_model(df):
    print("\n" + "=" * 50)
    print("🤖 LANGKAH 5: Model Prediksi")
    print("=" * 50)
    
    # Tambahkan indikator baru sebagai fitur untuk ML
    features = ['MA50', 'MA200', 'MA_Diff', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'close_price']
    df_clean = df.dropna(subset=features + ['Signal'])
    
    if len(df_clean) < 10 or df_clean['Signal'].sum() == 0:
        print("⚠️ Data tidak cukup (atau tidak ada Golden Cross) untuk melatih model.")
        return None
        
    X = df_clean[features].values
    y = df_clean['Signal'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model Random Forest selesai dilatih (Akurasi: {acc*100:.2f}%)")
    return model


# ==========================================
# LANGKAH 6: Visualisasi Interaktif (Plotly)
# ==========================================
def plot_interactive_chart(df, target_saham='ADRO', save_path='golden_cross_interactive.html'):
    print("\n" + "=" * 50)
    print("📉 LANGKAH 6: Visualisasi Interaktif")
    print("=" * 50)
    
    # Buat figure
    fig = go.Figure()

    # 1. Plot Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open_price'], high=df['high_price'],
        low=df['low_price'], close=df['close_price'],
        name='Harga (OHLC)',
        increasing_line_color='#00ff88', decreasing_line_color='#ff6b6b'
    ))

    # 2. Plot MA50 dan MA200
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='#00d2ff', width=2), name='MA50 (Pendek)'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='#ffaa00', width=2), name='MA200 (Panjang)'))

    # 3. Plot Bollinger Bands (Opsional, di-set transparansi rendah agar tidak berantakan)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='BB Upper', opacity=0.5))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='BB Lower', opacity=0.5))

    # 4. Tandai Golden Cross
    gc = df[df['Signal'] == 1]
    if len(gc) > 0:
        fig.add_trace(go.Scatter(
            x=gc.index, y=gc['close_price'],
            mode='markers+text',
            marker=dict(symbol='triangle-up', color='magenta', size=20, line=dict(color='white', width=2)),
            text=["GC"] * len(gc),
            textposition="top center",
            textfont=dict(color="magenta", size=14, family="Arial Black"),
            name='Golden Cross'
        ))

    # Styling dan Layout
    fig.update_layout(
        title=f'📈 Analisis Teknikal Interaktif — {target_saham.upper()}',
        yaxis_title='Harga (IDR)',
        xaxis_title='Tanggal',
        template='plotly_dark',
        xaxis_rangeslider_visible=True, # Menampilkan slider waktu di bawah
        hovermode='x unified',          # Menampilkan semua data saat di-hover pada tanggal tertentu
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Simpan sebagai file HTML
    fig.write_html(save_path)
    
    print(f"✅ Grafik INTERAKTIF tersimpan sebagai: {save_path}")
    print(f"👉 SILAKAN BUKA FILE '{save_path}' MENGGUNAKAN BROWSER (Chrome/Safari/Edge) UNTUK MELIHATNYA!")
    print()


# ==========================================
# EKSEKUSI PIPELINE
# ==========================================
if __name__ == "__main__":
    FILE_PATH = 'data/data_sample.csv' # Pastikan menggunakan file simulasi 4 tahun yang dibuat sebelumnya
    TARGET_SAHAM = 'adro'
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ File '{FILE_PATH}' tidak ditemukan. Pastikan path-nya benar.")
    else:
        df_raw = load_data(FILE_PATH)
        df_clean = preprocess(df_raw, target_saham=TARGET_SAHAM)
        
        # Menggunakan fungsi baru untuk indikator teknikal lengkap
        df_features = add_technical_indicators(df_clean)
        
        df_signals = detect_golden_cross(df_features)
        model = build_model(df_signals)
        
        # Menghasilkan output HTML interaktif
        plot_interactive_chart(df_signals, target_saham=TARGET_SAHAM)