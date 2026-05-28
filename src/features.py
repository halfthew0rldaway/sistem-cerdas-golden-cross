"""
Langkah 3: Feature Engineering
Menghitung Moving Average 50 dan 200 hari.
"""
import pandas as pd


def add_moving_averages(df, short_window=50, long_window=200):
    """
    Hitung dan tambahkan kolom Moving Average ke dataframe.
    
    Parameters:
        df (pd.DataFrame): DataFrame dengan kolom 'Close'
        short_window (int): Periode MA pendek (default: 50)
        long_window (int): Periode MA panjang (default: 200)
        
    Returns:
        pd.DataFrame: DataFrame dengan kolom MA50 dan MA200 ditambahkan
    """
    print("=" * 50)
    print("📊 LANGKAH 3: Feature Engineering (Moving Averages)")
    print("=" * 50)
    
    df = df.copy()
    
    # Hitung MA50 (rata-rata harga 50 hari terakhir)
    df['MA50'] = df['Close'].rolling(window=short_window).mean()
    print(f"\n✅ MA{short_window} dihitung (rata-rata {short_window} hari)")
    
    # Hitung MA200 (rata-rata harga 200 hari terakhir)
    df['MA200'] = df['Close'].rolling(window=long_window).mean()
    print(f"✅ MA{long_window} dihitung (rata-rata {long_window} hari)")
    
    # Hitung selisih MA (fitur tambahan untuk model)
    df['MA_Diff'] = df['MA50'] - df['MA200']
    print(f"✅ MA_Diff dihitung (MA50 - MA200)")
    
    # Info NaN karena rolling window
    nan_count = df['MA200'].isna().sum()
    print(f"\n⚠️  {nan_count} baris pertama belum punya MA200 (butuh minimal {long_window} hari data)")
    print(f"   Baris valid: {len(df) - nan_count} dari {len(df)}")
    print()
    
    return df


if __name__ == "__main__":
    from load_data import load_data
    from preprocess import preprocess
    
    df = load_data()
    df = preprocess(df)
    df = add_moving_averages(df)
    print(df[['Close', 'MA50', 'MA200', 'MA_Diff']].tail(10))
