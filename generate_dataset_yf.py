import yfinance as yf
import pandas as pd
import datetime

def main():
    print("=" * 60)
    print("📥 GENERATOR DATASET SAHAM DARI YFINANCE (ETL PIPELINE)")
    print("=" * 60)
    
    # =========================================================================
    # LANGKAH 1: INISIALISASI DAFTAR EMITEN DAN RENTANG WAKTU (PARAMETER)
    # =========================================================================
    # Daftar 45 saham berkapitalisasi besar (Indeks LQ45) sebagai representasi pasar
    emiten_list = [
        'BMTR.JK', 'ADRO.JK', 'AALI.JK', 'BBCA.JK', 'BBRI.JK', 'TLKM.JK', 'GOTO.JK', 'ASII.JK',
        'BMRI.JK', 'BBNI.JK', 'AMMN.JK', 'TPIA.JK', 'BYAN.JK', 'BREN.JK', 'ICBP.JK', 'UNVR.JK',
        'KLBF.JK', 'AMRT.JK', 'CPIN.JK', 'INDF.JK', 'PGEO.JK', 'BRPT.JK', 'MDKA.JK', 'AKRA.JK',
        'INKP.JK', 'PTBA.JK', 'ITMG.JK', 'UNTR.JK', 'SMGR.JK', 'BRIS.JK', 'PGAS.JK', 'MEDC.JK',
        'ARTO.JK', 'INCO.JK', 'ANTM.JK', 'HRUM.JK', 'ESSA.JK', 'MYOR.JK', 'BUKA.JK', 'MAPI.JK',
        'ACES.JK', 'CTRA.JK', 'BSDE.JK', 'PWON.JK', 'SMRA.JK'
    ]
    
    # Menetapkan rentang waktu 3 tahun untuk optimalisasi perhitungan Moving Average (MA200)
    start_date = '2023-06-01'
    end_date = '2026-06-11'
    print(f"[*] Menyiapkan ekstraksi {len(emiten_list)} emiten dari {start_date} hingga {end_date}...\n")
    
    # =========================================================================
    # LANGKAH 2: EKSTRAKSI DATA DARI SUMBER EKSTERNAL (EXTRACT)
    # =========================================================================
    df_list = []
    for emiten in emiten_list:
        try:
            print(f"[*] Mengekstraksi data historis untuk: {emiten}")
            # Request data harian menggunakan library yfinance
            df_emiten = yf.download(emiten, start=start_date, end=end_date, progress=False)
            
            if df_emiten.empty:
                print(f"    ⚠️ Peringatan: Data {emiten} tidak ditemukan di server.")
                continue
                
            # =========================================================================
            # LANGKAH 3: PEMBERSIHAN DAN TRANSFORMASI DATA (TRANSFORM)
            # =========================================================================
            # Meratakan format MultiIndex bawaan yfinance agar berbentuk tabel standar
            if isinstance(df_emiten.columns, pd.MultiIndex):
                df_emiten.columns = df_emiten.columns.droplevel(1)
                
            df_emiten = df_emiten.reset_index()
            df_emiten = df_emiten.dropna(subset=['Date']) # Menangani missing value pada dimensi waktu
            
            # Ekstraksi kode murni tanpa suffix market (misal: 'BMTR.JK' -> 'bmtr')
            kode_saham = emiten.split('.')[0].lower()
            
            # Normalisasi kolom agar sesuai dengan skema (schema) dataset target / template awal
            df_format = pd.DataFrame({
                'tanggal': df_emiten['Date'].dt.strftime('%Y-%m-%d'),
                'kode': kode_saham,
                'open_price': df_emiten['Open'],
                'high_price': df_emiten['High'],
                'low_price': df_emiten['Low'],
                'close_price': df_emiten['Close'],
                'volume': df_emiten['Volume']
            })
            
            # Menyimpan hasil transformasi sementara ke dalam list (RAM)
            df_list.append(df_format)
            
        except Exception as e:
            print(f"    ❌ Gagal mengekstraksi {emiten}: {e}")

    # =========================================================================
    # LANGKAH 4: PENGGABUNGAN DAN PENYIMPANAN DATA (LOAD)
    # =========================================================================
    if df_list:
        print("\n[*] Menggabungkan (Concatenation) seluruh blok data...")
        df_all = pd.concat(df_list, ignore_index=True)
        
        # Mengurutkan baris secara kronologis waktu dan alfabetikal kode saham
        df_all = df_all.sort_values(by=['tanggal', 'kode']).reset_index(drop=True)
        
        # Mengekspor (Load) DataFrame ke bentuk fisik file CSV
        output_filename = 'data_tester/dataset_yfinance_gabungan.csv'
        df_all.to_csv(output_filename, sep=';', index=False)
        
        # =========================================================================
        # LANGKAH 5: SUMMARY REPORT
        # =========================================================================
        print("\n" + "=" * 60)
        print("✅ PIPELINE ETL SELESAI DENGAN SUKSES!")
        print(f" Total Observasi Terkumpul : {len(df_all)} baris data valid.")
        print(f" Target File CSV Disimpan  : {output_filename}")
        print("=" * 60)
        
        print("\n💡 Petunjuk Eksekusi Model Machine Learning:")
        print(f"   python main_golden_cross.py --file {output_filename}")
    else:
        print("\n❌ Gagal membangun dataset. Harap periksa koneksi internet ke server Yahoo Finance.")

if __name__ == "__main__":
    main()
