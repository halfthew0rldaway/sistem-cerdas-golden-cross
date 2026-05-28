"""
Langkah 2: Preprocessing Data
Membersihkan dan menyiapkan data untuk analisis.
"""
import pandas as pd


def preprocess(df):
    """
    Preprocessing data saham:
    1. Ubah kolom tanggal ke format datetime
    2. Urutkan dari tanggal paling lama ke baru
    3. Hapus baris kosong (NaN)
    4. Set tanggal sebagai index
    
    Parameters:
        df (pd.DataFrame): DataFrame mentah dari load_data
        
    Returns:
        pd.DataFrame: DataFrame yang sudah bersih
    """
    print("=" * 50)
    print("🧹 LANGKAH 2: Preprocessing")
    print("=" * 50)
    
    # Buat copy agar tidak mengubah data asli
    df = df.copy()
    
    # 1. Ubah kolom Date ke datetime
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"\n✅ Kolom 'Date' diubah ke format datetime")
    
    # 2. Urutkan berdasarkan tanggal
    df = df.sort_values('Date')
    print(f"✅ Data diurutkan berdasarkan tanggal (lama → baru)")
    
    # 3. Hapus baris kosong
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    dropped = rows_before - rows_after
    if dropped > 0:
        print(f"⚠️  {dropped} baris kosong dihapus")
    else:
        print(f"✅ Tidak ada baris kosong")
    
    # 4. Set Date sebagai index
    df = df.set_index('Date')
    print(f"✅ Kolom 'Date' dijadikan index")
    
    # Info akhir
    print(f"\n📊 Data setelah preprocessing:")
    print(f"   Periode: {df.index[0].strftime('%Y-%m-%d')} s/d {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Jumlah baris: {len(df)}")
    print(f"   Kolom: {list(df.columns)}")
    print()
    
    return df


if __name__ == "__main__":
    from load_data import load_data
    df = load_data()
    df = preprocess(df)
    print(df.head())
