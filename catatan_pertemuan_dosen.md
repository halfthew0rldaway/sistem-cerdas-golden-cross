# 📝 Catatan Progress & Daftar Pertanyaan untuk Dosen

Dokumen ini berisi rangkuman progress yang bisa kamu sampaikan ke dosen besok, beserta daftar pertanyaan kritis terkait kendala data dan pemodelan Machine Learning untuk tugas **Sistem Cerdas - Deteksi Golden Cross**.

---

## 🚀 1. Laporan Progress (Yang Bisa Dilaporkan ke Dosen)

Saat ditanya sampai mana progressnya, kamu bisa sampaikan poin-poin berikut:

1. **Pipeline Program Sudah Selesai 100%:** Kami sudah membuat alur program lengkap (End-to-End) menggunakan Python. Alurnya meliputi:
   - **Load Data:** Membaca data CSV mentah.
   - **Preprocessing:** Membersihkan data dan memfilter berdasarkan kode saham tertentu.
   - **Feature Engineering:** Menghitung indikator teknikal secara otomatis (MA50, MA200, RSI, MACD, dan Bollinger Bands).
   - **Deteksi Pola:** Logika mendeteksi *Golden Cross* (saat MA50 memotong ke atas MA200).
   - **Pemodelan (Machine Learning):** Menggunakan algoritma **Random Forest Classifier** dari `scikit-learn` untuk memprediksi sinyal.
2. **Visualisasi Interaktif:** Kami tidak hanya menampilkan teks, tapi model kami menghasilkan **Grafik Candlestick Interaktif** (menggunakan library `Plotly`). Grafik ini bisa di-zoom, digeser, dan menunjukkan titik tepat di mana *Golden Cross* terjadi.
3. **Fleksibilitas Input:** Program kami sudah dinamis. Berapapun dan apapun file CSV yang nanti dikirimkan oleh bapak/ibu dosen, program kami tinggal dieksekusi lewat *command line* tanpa perlu merombak *source code* lagi.

---

## ❓ 2. Daftar Pertanyaan Kritis untuk Dosen (Yang Membingungkan / Butuh Klarifikasi)

Ini adalah pertanyaan-pertanyaan berbobot yang menunjukkan bahwa kelompokmu benar-benar mengerjakan dan memahami *domain knowledge* (ilmu trading) serta algoritma Machine Learning-nya.

### A. Pertanyaan Terkait Keterbatasan Data (Sangat Penting!)
> **Fakta Temuan Kami:** Berdasarkan pengecekan pada data `transaksi_harian_202606082112.csv`, data tersebut berisi transaksi dari tanggal **2 Januari 2026 sampai 5 Juni 2026**. Ini menghasilkan tepat **88 hari perdagangan aktif (unik)**. Sementara itu, *Golden Cross* mensyaratkan perhitungan MA200 (Rata-rata 200 hari ke belakang), yang butuh minimal 200 hari kerja (sekitar 1 tahun kalender).
*   **Pertanyaan:** *"Pak/Bu, terkait data CSV yang diberikan, kami mencoba memfilter satu saham (contoh: ADRO) dan setelah divalidasi, rentang datanya hanya dari 2 Jan - 5 Jun 2026 (sekitar 88 hari aktif). Karena pola Golden Cross secara matematis butuh perhitungan MA200 (200 hari), pola ini tidak bisa terdeteksi/dihitung sama sekali jika datanya kurang dari setahun. Apakah nanti data final untuk penilaian memiliki rentang waktu yang lebih panjang (misal 3-5 tahun)? Atau apakah kami boleh scraping data historis sendiri dari sumber luar (seperti Yahoo Finance) sebagai data latih agar model ML-nya bisa bekerja?"*

### B. Pertanyaan Terkait Tujuan Model Machine Learning
> **Konteks:** Menentukan *Golden Cross* sebenarnya hanya butuh logika matematika sederhana (IF MA50 > MA200). Jika disuruh membuat "Model dan Prediksi" menggunakan Machine Learning, kita harus memastikan apa yang sebenarnya ingin diprediksi.
*   **Pertanyaan:** *"Terkait kriteria output 'Model dan Prediksi', saat ini kami menggunakan algoritma Random Forest Classifier. Namun, karena Golden Cross bisa dicari dengan rumus pasti, kami ingin memastikan ekspektasi pemodelannya. Apakah Machine Learning ini ditujukan untuk memprediksi **kapan** Golden Cross akan terjadi di masa depan (forecasting hari)? Ataukah memprediksi **apakah harga saham benar-benar akan naik/profit** setelah Golden Cross terjadi (mengurangi sinyal palsu / false breakout)?"*

### C. Pertanyaan Terkait Kriteria Penilaian & UTS (Bab 1-3)
> **Konteks:** Soal nomor 4 menyebutkan UTS akan berisi Bab 1-3 penelitian.
*   **Pertanyaan:** *"Untuk laporan UTS Bab 1-3 penelitian, apakah fokus utama penulisan latar belakang dan rumusannya lebih condong ke arah analisis finansialnya (pola saham), atau difokuskan pada perbandingan kinerja algoritma Sistem Cerdasnya (misal: optimasi akurasi algoritma Random Forest)? Apakah ada format *paper* khusus yang harus kami ikuti (misal format IEEE)? "*

### D. Penjelasan Jika Dosen Bertanya: "Kenapa Akurasinya Bisa 98% - 99%? (Too Good To Be True)"
> **Konteks:** Di dunia Machine Learning, akurasi nyaris 100% sering disebut *Data Leakage* (bocoran data) atau akibat *Class Imbalance* (ketimpangan data).
*   **Penjelasan (Argumenmu):** *"Betul Pak/Bu, kami juga menyadari hal ini. Akurasi 99% ini sebenarnya terjadi karena dua alasan teknis yang wajar di tahap awal ini:*
    *   *Pertama, **Imbalanced Data**: Dalam 1.200 hari (5 tahun), momen Golden Cross itu sangat langka (cuma 2 kali). Artinya, ada 1.198 hari BUKAN Golden Cross. Kalau model ML cuma menebak 'Bukan Golden Cross' setiap hari secara buta, akurasinya sudah otomatis 99.8%.*
    *   *Kedua, **Data Leakage (Bocor)**: Variabel target (Signal Golden Cross) kan didapat dari rumus `MA50 > MA200`. Nah, karena `MA50` dan `MA200` juga kami jadikan 'Fitur Input' untuk AI, maka AI dengan sangat mudah 'menjiplak' rumus tersebut alih-alih memprediksi dari indikator lain (seperti RSI/MACD). Sebagai perbaikan ke depannya, kami berencana memodifikasi fitur inputnya agar lebih menantang bagi model."*

### E. Penjelasan: Kami Sangat Menghindari "False Positive" (Sinyal Palsu)
> **Konteks:** Dosen seringkali mencari sistem trading yang tidak mudah tertipu (false signal).
*   **Penjelasan (Argumenmu):** *"Untuk membuat model ini benar-benar bisa dipakai secara nyata (realistis), kami menerapkan aturan **STRICT PRECISION (Sangat Konservatif)**. Kami mematikan auto-balance kelas, dan memaksa model untuk memprediksi 'Golden Cross' HANYA jika ia merasa YAKIN **lebih dari 85%** (`predict_proba >= 0.85`). Ini dilakukan agar trader/pengguna aplikasi kami terhindar dari *False Positive* (AI salah menyuruh beli saham), karena dalam dunia trading lebih baik ketinggalan sinyal daripada kena sinyal palsu yang merugikan."*

---

## 💡 3. Tips Tambahan Saat Presentasi/Diskusi
- Tunjukkan file HTML (`golden_cross_interactive.html`) kepada dosen untuk memberikan kesan "*Wow*". Ini membuktikan bahwa kriteria **(b) Running model** sudah sangat siap.
- Jika ditanya **(c) Algoritma yang digunakan**, jawablah: *"Kami menggunakan **Random Forest Classifier**. Alasannya, Random Forest sangat tangguh menangani data finansial yang banyak *noise* (fluktuasi tidak menentu) dan mampu melihat interaksi antar berbagai fitur indikator teknikal (RSI, MACD, MA) secara bersamaan tanpa mudah mengalami *overfitting*."*

---

## 📌 Referensi: Deskripsi Soal Asli (Untuk Future AI Context)
Bagian ini menyimpan soal asli dari dosen sebagai *context memory* apabila ada pengembangan tugas di masa depan.
**Kelompok mengambil Use Case 2: Golden Cross.**

```text
1. Buat kelompok yang isinya 3 orang
2. Buat program dengan menggunakan bahasa python. Pilih :
    Use case 1 :
        Ingestion (2 kelompok) -> Memasukkan data dari web / scrap kedalam database. 
            Input   : API / Website IDX (https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=9999&start=0) / Atau sumber lain
            Output  : Database
    Use case 2 : -> Melakukan modelling dari data yang sudah ada. Tujuannya untuk mendeteksi pola dibawah ini
        Head & Shoulders (2 kelompok)
        Inverted H&S (2 kelompok)
        Double top / bottom (2 kelompok)
        Golden cross (2 kelompok)
        Death cross (2 kelompok)
        Breakout (2 kelompok)
        Support/resistance breakout (2 kelompok)
            Input   : Data csv (nanti akan saya kirimkan datanya)
            Output  : Model dan Prediksi
3.  Kriteria penilaian, 
    a. Penjelasan model yang dibuat
    b. Running model dan akurasi
    c. Algoritma yang digunakan beserta flow eksekusi
4.  Untuk soal UTS : akan berisi bab 1-3 penelitian
5.  Bisa pakai library python apa saja dengan penjelasan yang nanti kalian harus mengerti
```
