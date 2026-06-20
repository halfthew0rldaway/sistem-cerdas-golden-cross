#  Deteksi Golden Cross dengan Machine Learning

**Tugas Kelompok — Sistem Cerdas (Use Case 2: Pattern Detection)**

Proyek ini adalah sistem kecerdasan buatan berbasis *Machine Learning* yang dirancang dengan **standar industri (Best Practice)** untuk memprediksi kemunculan pola *Golden Cross* pada pergerakan saham.

---

##  Stack Teknologi (Tech Stack)
Proyek ini sepenuhnya dibangun menggunakan **Python** dengan library pendukung berikut:
- **Pandas & NumPy:** Untuk *Data Wrangling* (membaca CSV, manipulasi, dan kalkulasi matematika).
- **Scikit-Learn:** Pustaka utama untuk Machine Learning (pemodelan dan evaluasi).
- **Imbalanced-Learn (`imblearn`):** Pustaka khusus untuk menyeimbangkan data (*Undersampling*).
- **Plotly:** Untuk menggambar visualisasi grafik saham yang interaktif (bisa di-zoom/di-geser di browser HTML).

---

##  Fitur Utama & Perbandingan dengan Versi Lama (`old-ver`)

Proyek ini telah dirombak drastis dari versi lama (`old-ver`) untuk menyelesaikan masalah akurasi yang "terlalu sempurna" (99%-100%).

### Beda Mekanisme: Versi Baru vs Versi Lama (`old-ver`)
1. **Pencegahan Data Leakage (Bocoran Data):**
   - **Versi Lama:** Model diberikan data garis `MA50` dan `MA200` hari ini, lalu diminta menebak *Golden Cross* hari ini. Karena *Golden Cross* secara matematis *memang* persilangan kedua garis itu, AI hanya "menjiplak" rumus tersebut alih-alih belajar, menghasilkan tebakan 100% yang palsu.
   - **Versi Baru:** Garis MA **dilarang** masuk ke dalam otak AI (fitur). AI hanya diberikan indikator momentum (RSI, MACD, Bollinger Bands) hari ini, dan dipaksa memprediksi kejadian *Golden Cross* untuk **BESOK (H+1)**. Ini namanya *Forecasting* sejati!
2. **Handling Imbalanced Data:**
   - **Versi Lama:** Kejadian *Golden Cross* sangat langka (cuma ~1% dari total data). Karena tidak diselaraskan, AI versi lama menjadi "malas" dan selalu menebak "Tidak Ada Golden Cross" setiap hari. Hasilnya? Akurasi 99% karena 99% hari memang tidak ada *Golden Cross*.
   - **Versi Baru:** Menggunakan fungsi `RandomUnderSampler`, data diseimbangkan secara paksa (*Undersampling*), sehingga model benar-benar "bersusah payah" belajar mengenali pola secara nyata.
3. **Hasil:** Akurasi versi baru berada di rentang **~82%**. Dalam dunia deteksi anomali finansial, ini adalah angka yang sangat realistis dan istimewa!

---

##  Algoritma dan Komparasi
Model ini menggunakan algoritma **Random Forest Classifier**.

**Kenapa Random Forest?**
Data saham penuh dengan *noise* (fluktuasi sesaat yang mengecoh). Random Forest sangat tangguh dalam menyaring *noise* ini karena ia bekerja dengan membuat ratusan "Pohon Keputusan" dan mengambil suara terbanyak. Ia juga pintar melihat interaksi antar beberapa indikator teknikal (RSI, MACD) secara bersamaan.

**Komparasi dengan Algoritma Lain:**
Selama fase eksperimen (pipeline analysis), kami sempat membandingkannya dengan beberapa algoritma standar:
- **SVM (Support Vector Machine):** Memakan waktu terlalu lama saat datanya bertambah besar dan kesulitan menangkap hubungan non-linear dari indikator kompleks.
- **Decision Tree (Pohon Keputusan Tunggal):** Sangat rentan mengalami *overfitting* (terlalu hafal dengan data pelatihan masa lalu, tapi gagal memprediksi data masa depan).
- **Logistic Regression:** Algoritma linear ini terlalu sederhana dan kurang bisa menangkap pola rumit pergerakan harga saham.
Oleh karena itu, **Random Forest** terpilih sebagai model yang paling stabil dan akurat secara *default*.

---

##  Bagaimana Sistem Ini Bekerja? (Flow Step-by-Step)

Alur kerja program berjalan berurutan sesuai kaidah *Machine Learning Lifecycle*:

1. **Memuat Data (Data Ingestion):** Memuat file CSV transaksi saham sesuai input dari pengguna.
2. **Feature Engineering:** Menghitung rumus matematika secara otomatis untuk Indikator Teknikal (Return, RSI, MACD, dan Posisi Bollinger Bands).
3. **Mendefinisikan Target:** Program mencari letak pasti *Golden Cross* hari ini, lalu "menggeser" target ke belakang (H+1) agar menjadi target prediksi untuk esok hari.
4. **Data Splitting (Komposisi Pelatihan):** Data dipotong menjadi dua bagian. **80% untuk Latihan (Training)** dan **20% untuk Ujian (Testing)**.
5. **Standardisasi & Undersampling:** Menyamakan skala angka (misal, harga vs persentase desimal) menggunakan `StandardScaler`, dan menyeimbangkan jumlah kelas menggunakan `RandomUnderSampler`.
6. **Pelatihan (Training):** Model Random Forest dilatih dan langsung diuji akurasinya.
7. **Visualisasi:** Model meng-ekspor hasil ke file HTML interaktif.

---

##  Struktur File Proyek

- `main_golden_cross.py` : **File Program Utama.**
- `download_yahoo_data.py` : Script pendamping untuk mengunduh data langsung dari Yahoo Finance jika kamu butuh data lain.
- `requirements.txt` : Daftar *library* Python yang dibutuhkan.
- `old-ver/` : Folder arsip yang menyimpan mekanisme lama.

---

##  Cara Menjalankan Program

### 1. Instalasi Library
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Program Utama
Jalankan file dengan menunjuk CSV yang ingin dianalisis (Contoh menggunakan saham BMTR):
```bash
python main_golden_cross.py --file "transaksi_harian_202606130928(daridosenbarukhususgoldencross).csv" --saham bmtr
```

---

##  Memahami Hasil Output & Grafik

1. **Hasil Terminal:** Menampilkan matriks evaluasi seperti *Accuracy* (~82%) dan *ROC-AUC Score*.
2. **File HTML:** File bernama `golden_cross_interactive.html` akan muncul. Klik dua kali (atau buka di Chrome) untuk melihat grafik saham interaktif dan letak *Golden Cross*!
