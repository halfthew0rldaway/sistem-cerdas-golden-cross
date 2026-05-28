"""
Langkah 4: Deteksi Golden Cross
Mendeteksi kapan MA50 memotong MA200 dari bawah ke atas.
"""
import pandas as pd


def detect_golden_cross(df):
    """
    Deteksi Golden Cross: MA50 melewati MA200 dari bawah ke atas.
    
    Logika:
    - Hari ini: MA50 > MA200
    - Kemarin: MA50 <= MA200
    - Jika keduanya terpenuhi → Golden Cross!
    
    Parameters:
        df (pd.DataFrame): DataFrame dengan kolom MA50 dan MA200
        
    Returns:
        pd.DataFrame: DataFrame dengan kolom 'Signal' ditambahkan
    """
    print("=" * 50)
    print("🔍 LANGKAH 4: Deteksi Golden Cross")
    print("=" * 50)
    
    df = df.copy()
    
    # Inisialisasi kolom sinyal
    df['Signal'] = 0
    
    # Deteksi Golden Cross
    for i in range(1, len(df)):
        ma50_today = df['MA50'].iloc[i]
        ma200_today = df['MA200'].iloc[i]
        ma50_yesterday = df['MA50'].iloc[i - 1]
        ma200_yesterday = df['MA200'].iloc[i - 1]
        
        # Skip jika ada NaN
        if pd.isna(ma50_today) or pd.isna(ma200_today) or \
           pd.isna(ma50_yesterday) or pd.isna(ma200_yesterday):
            continue
        
        # Cek apakah MA50 baru saja melewati MA200 dari bawah
        if ma50_today > ma200_today and ma50_yesterday <= ma200_yesterday:
            df.iloc[i, df.columns.get_loc('Signal')] = 1  # Golden Cross!
    
    # Tampilkan hasil
    golden_cross_dates = df[df['Signal'] == 1].index
    n_gc = len(golden_cross_dates)
    
    print(f"\n🎯 Jumlah Golden Cross terdeteksi: {n_gc}")
    
    if n_gc > 0:
        print(f"\n📅 Tanggal-tanggal Golden Cross:")
        for i, date in enumerate(golden_cross_dates, 1):
            close_price = df.loc[date, 'Close']
            print(f"   {i}. {date.strftime('%Y-%m-%d')} — Harga Close: {close_price:.2f}")
    else:
        print("   ⚠️  Tidak ada Golden Cross terdeteksi dalam periode ini.")
    
    print()
    return df


if __name__ == "__main__":
    from load_data import load_data
    from preprocess import preprocess
    from features import add_moving_averages
    
    df = load_data()
    df = preprocess(df)
    df = add_moving_averages(df)
    df = detect_golden_cross(df)
