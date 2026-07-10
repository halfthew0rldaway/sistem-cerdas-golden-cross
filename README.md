# Deteksi Golden Cross dengan Machine Learning

**Proyek Sistem Cerdas (Use Case: Pattern Detection)**

Proyek ini merupakan implementasi sistem kecerdasan buatan berbasis *Machine Learning* yang dirancang mengikuti standar industri (Best Practice) untuk memprediksi kemunculan pola *Golden Cross* pada pergerakan pasar saham.

## Deskripsi Mekanisme Sistem

Sistem ini didesain secara khusus untuk mencegah terjadinya kebocoran data (*Data Leakage*) dan menangani persoalan ketidakseimbangan kelas (*Imbalanced Data*) melalui mekanisme berikut:

1. **Pencegahan Data Leakage (Bocoran Data):**
   Model diarahkan untuk memprediksi kejadian *Golden Cross* pada H+1 (esok hari) dengan mempelajari berbagai fitur teknikal hari ini seperti indikator momentum (RSI, MACD, *Bollinger Bands*, dan sebagainya). Hal ini memastikan sistem belajar menangkap pola fundamental (*Forecasting* sesungguhnya) tanpa mengintip langsung kalkulasi *Moving Average* di masa depan yang secara harafiah membentuk *Golden Cross*.

2. **Penanganan Imbalanced Data:**
   Insiden *Golden Cross* tergolong langka (bersifat anomali) dibandingkan hari biasa tanpa persilangan. Agar model terhindar dari bias mayoritas (selalu menebak tidak ada *Golden Cross*), proses *Undersampling* diterapkan dengan metode `RandomUnderSampler`. Hal ini memungkinkan penyetaraan jumlah data latih antara kelas positif dan negatif sehingga model terpaksa belajar mengenali pola teknikal yang akurat.

3. **Algoritma Klasifikasi:**
   Sistem memanfaatkan algoritma *Random Forest Classifier*. Algoritma ansambel ini dipilih berdasarkan kemampuannya yang stabil dalam meredam *noise* pada data keuangan, serta menghindari *overfitting* yang seringkali ditemukan pada pemodelan *Decision Tree* tunggal atau algoritma linier seperti *Logistic Regression*.

## Alur Kerja Sistem (Pipeline)

Pemrosesan *Machine Learning Lifecycle* berjalan otomatis dengan tahapan sebagai berikut:

1. **Data Ingestion:** Pemuatan riwayat harga saham (berformat `.csv`) yang dikelola di dalam direktori `data_tester/`.
2. **Feature Engineering:** Kalkulasi rumusan indikator teknikal (Return, RSI, MACD, Stochastic, ATR, dan posisi *Bollinger Bands*) dieksekusi untuk merepresentasikan kondisi fundamental harga.
3. **Definisi Target Prediksi:** Label *Golden Cross* diidentifikasi pada hari berjalan (T) kemudian digeser sebagai sasaran tebakan pada hari sebelumnya (T-1) untuk memfasilitasi obyektif peramalan.
4. **Data Splitting & Preprocessing:** Pembagian dataset menjadi proporsi *Training*, *Validation*, dan *Testing*. Data dilanjutkan menuju tahap normalisasi (*StandardScaler*) dan penyeimbangan (*Undersampling*).
5. **Pelatihan & Evaluasi Model:** Evaluasi dilakukan menggunakan model yang dilatih terhadap beberapa skenario validasi, mencakup uji *out-of-time* dan uji fluktuasi (*noise*).
6. **Visualisasi Interaktif:** Pembuatan representasi visual otomatis dengan *Plotly* yang dapat ditinjau melalui peramban.

## Struktur Repositori

- `main_golden_cross.py` : Berkas inti implementasi *Machine Learning Pipeline*.
- `compare_models.py` : Skrip pengujian dan perbandingan rasio keberhasilan beberapa algoritma model klasifikasi.
- `app.py` : Program peladen HTTP berbasis *Flask* untuk merender visualisasi panel kendali (*Dashboard*).
- `download_yahoo_data.py` & `generate_dataset_yf.py` : Modul ekstraksi yang difungsikan untuk menghimpun data emiten dari ekosistem eksternal.
- `data_tester/` : Direktori sentral yang mewadahi seluruh dataset masukan riwayat transaksi saham.
- `dashboard/` : Direktori yang menyimpan elemen *Front-end* (HTML, CSS, dan JS) dari antarmuka Web.

## Panduan Instalasi dan Menjalankan Sistem

Prasyarat sebelum memulai instalasi, pastikan lingkungan *Python 3.x* telah diunduh dan dipersiapkan dengan baik.

### 1. Instalasi Dependensi Pustaka

Buka antarmuka baris perintah (*Terminal* atau *Command Prompt*) pada direktori akar proyek, kemudian eksekusi perintah berikut untuk menginstal seluruh pustaka prasyarat:

```bash
pip install -r requirements.txt
```

### 2. Eksekusi Analisis Prediksi (Pipeline Utama)

Untuk melatih model prediksi *Machine Learning* dan membuat berkas keluaran grafik yang interaktif, jalankan perintah berikut:

```bash
python main_golden_cross.py --file data_tester/daridosenbarukhususgoldencross.csv --saham bmtr
```

Proses tersebut akan menghasilkan laporan kinerja terminal beserta berkas luaran `golden_cross_interactive.html` di dalam direktori kerja.

### 3. Komparasi Kinerja Algoritma

Untuk membandingkan kapabilitas *Random Forest* dengan *Support Vector Machine*, *Decision Tree*, atau *Logistic Regression* terhadap dataset yang sama, eksekusi perintah di bawah ini:

```bash
python compare_models.py --file data_tester/daridosenbarukhususgoldencross.csv
```

### 4. Peluncuran Web Dashboard

Aplikasi memiliki integrasi berbasis antarmuka grafis Web. Jalankan layanan lokal menggunakan *Flask*:

```bash
python app.py
```
Akses tautan `http://127.0.0.1:5000` melalui peramban web untuk mengunggah dataset secara visual maupun melihat representasi interaktif.

---
Proyek siap untuk dilanjutkan ke tahap *Deployment* maupun pencadangan berbasis *Git* tanpa adanya jejak artefak yang memberatkan repositori.
