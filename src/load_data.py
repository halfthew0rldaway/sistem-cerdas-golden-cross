"""
Langkah 1: Load Data dari CSV
Membaca file CSV saham dan melakukan pengecekan awal.
"""
import pandas as pd
import os


def load_data(filepath='data/saham.csv'):
    """
    Baca file CSV saham menggunakan pandas.
    
    Parameters:
        filepath (str): Path ke file CSV
        
    Returns:
        pd.DataFrame: DataFrame berisi data saham
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File '{filepath}' tidak ditemukan!\n"
            f"Pastikan file CSV sudah ada di folder data/.\n"
            f"Atau jalankan: python data/generate_sample.py"
        )
    
    df = pd.read_csv(filepath)
    
    # Tampilkan info awal
    print("=" * 50)
    print("📥 LANGKAH 1: Load Data")
    print("=" * 50)
    print(f"\n📄 File: {filepath}")
    print(f"📊 Jumlah baris: {len(df)}")
    print(f"📋 Kolom: {list(df.columns)}")
    print(f"\n🔍 5 Baris Pertama:")
    print(df.head())
    print()
    
    # Validasi kolom yang dibutuhkan
    required_cols = ['Date', 'Close']
    for col in required_cols:
        if col not in df.columns:
            # Coba cari dengan case-insensitive
            found = [c for c in df.columns if c.lower() == col.lower()]
            if found:
                df = df.rename(columns={found[0]: col})
                print(f"⚠️  Kolom '{found[0]}' di-rename ke '{col}'")
            else:
                raise ValueError(f"Kolom '{col}' tidak ditemukan dalam data!")
    
    print("✅ Data berhasil dimuat!\n")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.info())
