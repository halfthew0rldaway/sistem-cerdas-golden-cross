import yfinance as yf
import pandas as pd
import datetime

# Saham Adaro Energy (ADRO) di Bursa Efek Indonesia kodenya ADRO.JK
stock_code = 'ADRO.JK'
print(f"Mengunduh data untuk {stock_code} dari Yahoo Finance...")

# Unduh data 5 tahun terakhir
df = yf.download(stock_code, period='5y')

if df.empty:
    print("Gagal mengunduh data.")
else:
    # yfinance terbaru mengembalikan MultiIndex columns (Price, Ticker), kita perlu flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # Reset index agar Date menjadi kolom
    df = df.reset_index()
    
    # Format sesuai dengan yang diharapkan program (huruf kecil semua)
    df_formatted = pd.DataFrame({
        'tanggal': df['Date'].dt.strftime('%Y-%m-%d'),
        'kode': 'adro', # hardcode kode saham
        'open_price': df['Open'],
        'high_price': df['High'],
        'low_price': df['Low'],
        'close_price': df['Close'],
        'volume': df['Volume']
    })
    
    # Simpan ke CSV dengan separator ;
    output_filename = 'data_adro_5_tahun.csv'
    df_formatted.to_csv(output_filename, sep=';', index=False)
    
    print(f"✅ Berhasil! Data disimpan di {output_filename}")
    print(f"Total baris data: {len(df_formatted)}")
    print("Sekarang kamu bisa menjalankan: python main_golden_cross.py --file data_adro_5_tahun.csv --saham adro")
