"""
SCRIPT PENGUNDUH DATA SAHAM (YAHOO FINANCE)
=========================================================
Script ini digunakan untuk mengunduh data historis saham
langsung dari server Yahoo Finance (yfinance) dan mengubah
formatnya agar cocok digunakan pada program Machine Learning kita.
=========================================================
"""

import yfinance as yf
import pandas as pd
import datetime

# 1. Menentukan Saham yang Ingin Diunduh
# Kode saham di Bursa Efek Indonesia (IDX) harus diakhiri dengan ".JK" di Yahoo Finance.
# Contoh: Adaro Energy (ADRO) menjadi ADRO.JK
stock_code = 'ADRO.JK'
print(f"🔄 Mengunduh data untuk saham '{stock_code}' dari server Yahoo Finance...")

# 2. Mengunduh Data
# Kita meminta data untuk 5 tahun terakhir ('5y').
# Opsi lain yang bisa digunakan: '1mo' (1 bulan), '1y' (1 tahun), 'max' (semua data dari awal).
df = yf.download(stock_code, period='5y')

if df.empty:
    print("❌ Gagal mengunduh data. Pastikan koneksi internet aktif atau kode saham benar.")
else:
    print("✅ Data berhasil diunduh. Sedang merapikan format kolom...")
    
    # 3. Merapikan Format Kolom dari yfinance
    # Kadang yfinance mengembalikan format kolom "MultiIndex" (berlapis).
    # Kita perlu meratakannya (flatten) agar mudah dibaca oleh program utama kita.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # Memindahkan kolom tanggal (yang tadinya menjadi Index) menjadi kolom biasa
    df = df.reset_index()
    
    # 4. Menyesuaikan Format Kolom dengan Program Utama
    # Program utama kita menggunakan huruf kecil ('open_price', 'close_price')
    df_formatted = pd.DataFrame({
        'tanggal': df['Date'].dt.strftime('%Y-%m-%d'), # Mengubah format tanggal menjadi YYYY-MM-DD
        'kode': 'adro',                                # Menandai data ini milik saham adro (huruf kecil)
        'open_price': df['Open'],
        'high_price': df['High'],
        'low_price': df['Low'],
        'close_price': df['Close'],
        'volume': df['Volume']
    })
    
    # 5. Menyimpan ke dalam File CSV
    # Sep=';' artinya kita memisahkan kolom dengan titik koma, bukan koma.
    # index=False agar nomor urut (0, 1, 2...) tidak ikut tersimpan ke dalam file.
    output_filename = 'data_adro_5_tahun.csv'
    df_formatted.to_csv(output_filename, sep=';', index=False)
    
    print(f"✅ Selesai! Data final berhasil disimpan dalam file: '{output_filename}'")
    print(f"📊 Total baris data yang didapat: {len(df_formatted)} hari perdagangan.")
    print("💡 TIPS: Sekarang kamu bisa mengetes Machine Learning menggunakan data 5 tahun ini dengan perintah:")
    print("   python main_golden_cross.py --file data_adro_5_tahun.csv --saham adro")

