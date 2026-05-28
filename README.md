# 📈 Golden Cross Detection

**Tugas Kelompok — Sistem Cerdas (Use Case 2: Pattern Detection)**

## Apa itu Golden Cross?

Golden Cross adalah sinyal teknikal dalam analisis saham yang terjadi ketika **Moving Average jangka pendek (MA50)** memotong ke atas **Moving Average jangka panjang (MA200)**.

Sinyal ini dianggap sebagai indikator **bullish** (harga akan naik) dan sering digunakan trader sebagai sinyal beli.

```
Sinyal BELI = MA50 melewati MA200 dari bawah ke atas ✅
```

## Kenapa Golden Cross Penting?

- Menunjukkan perubahan momentum dari bearish ke bullish
- Digunakan secara luas oleh trader dan analis teknikal
- Historis menunjukkan korelasi dengan kenaikan harga jangka menengah-panjang

## Library yang Digunakan

| Library | Kegunaan |
|---|---|
| `pandas` | Membaca dan mengolah data CSV (seperti Excel versi Python) |
| `numpy` | Operasi matematika dan array |
| `matplotlib` | Membuat grafik dan visualisasi |
| `scikit-learn` | Membuat dan mengevaluasi model machine learning |
| `seaborn` | Styling grafik yang lebih menarik |

## Struktur Project

```
golden_cross/
├── data/
│   ├── saham.csv              # Data CSV saham
│   └── generate_sample.py    # Generator data sample
├── src/
│   ├── load_data.py          # Langkah 1: Baca CSV
│   ├── preprocess.py         # Langkah 2: Bersihkan data
│   ├── features.py           # Langkah 3: Hitung MA50 & MA200
│   ├── detect.py             # Langkah 4: Deteksi Golden Cross
│   ├── model.py              # Langkah 5: Model prediksi
│   └── visualize.py          # Langkah 6: Visualisasi
├── main.py                   # File utama
├── requirements.txt          # Daftar library
└── README.md                 # Dokumentasi
```

## Cara Menjalankan

```bash
# 1. Install library
pip install -r requirements.txt

# 2. Generate data sample (jika belum punya CSV dari dosen)
cd data && python generate_sample.py && cd ..

# 3. Jalankan program utama
python main.py
```

## Alur Eksekusi (Flow)

```
Load CSV → Preprocessing → Hitung MA50 & MA200 → Deteksi Golden Cross → Model Prediksi → Visualisasi
```

1. **Load Data** — Baca file CSV, validasi kolom
2. **Preprocessing** — Konversi datetime, sort, hapus NaN, set index
3. **Feature Engineering** — Hitung MA50, MA200, dan selisihnya
4. **Deteksi** — Identifikasi titik-titik Golden Cross
5. **Model** — Random Forest Classifier untuk prediksi sinyal
6. **Visualisasi** — Plot grafik dengan penanda Golden Cross

## Model yang Digunakan

### Random Forest Classifier

**Kenapa dipilih:**
- Robust terhadap overfitting
- Bisa handle data imbalanced (Golden Cross jarang terjadi)
- Memberikan feature importance untuk interpretasi
- Tidak memerlukan banyak tuning hyperparameter

**Fitur yang digunakan:**
- MA50 (Moving Average 50 hari)
- MA200 (Moving Average 200 hari)
- MA_Diff (Selisih MA50 - MA200)
- Close (Harga penutupan)

## Output

- **Terminal**: Laporan lengkap setiap langkah + akurasi model
- **File**: `golden_cross_chart.png` — Grafik visualisasi
