# 📑 Laporan Perbaikan Mekanisme Model Machine Learning
**Topik: Deteksi Pola Golden Cross pada Pergerakan Saham**

Dokumen ini disusun sebagai ringkasan perubahan (Refactoring) yang telah dilakukan pada sistem Machine Learning dari versi awal (Old Version) menuju ke versi final yang saat ini berjalan. Tujuan utama dari perbaikan ini adalah menyelesaikan isu *Fake Accuracy* (Akurasi Palsu) 99%-100% dan menerapkan *Best Practice* standar industri.

---

## 1. Ringkasan Eksekutif (TL;DR)
Pada versi awal, model Machine Learning terlihat sempurna dengan akurasi hampir 100%. Namun, setelah dilakukan investigasi teknikal, hal ini terdeteksi sebagai anomali akibat **Data Leakage** (kebocoran kunci jawaban) dan **Imbalanced Data** (ketimpangan kelas). Pada versi terbaru, mekanisme ini dirombak total menjadi model *Forecasting* (peramalan) yang jauh lebih tangguh, objektif, dan realistis dengan tingkat akurasi ~82%.

---

## 2. Tabel Komparasi Mekanisme

| Aspek Analisis | Versi Lama (*Old Version*) | Versi Baru (*Final Version*) | Alasan Perubahan (Bahan Presentasi) |
| :--- | :--- | :--- | :--- |
| **Tujuan Prediksi** | Menebak *Golden Cross* **Hari Ini**. | Memprediksi *Golden Cross* **Besok (H+1)**. | Mengubah konsep dari sekadar "Kalkulator Rumus" menjadi model peramalan (*Forecasting*) masa depan yang sebenarnya. |
| **Input Fitur (X)** | Memasukkan nilai MA50 dan MA200 secara langsung ke dalam model. | **DILARANG** memasukkan MA. Hanya menggunakan indikator murni (RSI, MACD, Bollinger Bands). | **Mencegah Data Leakage.** Karena GC dihitung dari rumus MA, memasukkan MA membuat model sekadar "menjiplak" rumus (curang). Kini AI dipaksa mencari pola momentum murni. |
| **Keseimbangan Data** | Data dibiarkan alami (99% Bukan GC, 1% GC). | Menerapkan teknik **Undersampling** (`RandomUnderSampler`). | Kejadian GC sangat langka. Jika tidak diseimbangkan, AI menjadi "malas" dan sekadar menebak "Tidak" setiap hari untuk mendapat skor 99%. |
| **Tingkat Akurasi** | **99.9% - 100%** (Akurasi Palsu). | **~82% - 85%** dengan evaluasi tambahan ROC-AUC Score. | Skor 85% adalah skor realistis dan sangat bagus di dunia finansial, membuktikan AI benar-benar belajar menebak pola, bukan sekadar menghafal. |
| **Fleksibilitas Data** | File CSV dan nama saham di-*hardcode* di dalam skrip. | Sangat dinamis menggunakan *Command Line Arguments* (`argparse`). | Agar sistem bisa menerima data ujian apa pun (file CSV apa pun) yang diberikan dosen tanpa perlu mengubah baris kode *Python*. |

---

## 3. Detail Penjelasan Teknis (Untuk Q&A Dosen)

### A. Mengapa Akurasi 100% Bisa Terjadi di Versi Lama?
Ini adalah *jebakan* klasik di dunia *Data Science*. Akurasi 100% kemarin disebabkan oleh model yang secara tidak sengaja "melihat kunci jawaban". Variabel target (*Golden Cross*) ditentukan menggunakan logika `IF MA50 > MA200`. Karena di versi lama kita juga memasukkan data `MA50` dan `MA200` ke dalam variabel pengenal (fitur), maka *Machine Learning* hanya perlu membuat logika `IF` sederhana alih-alih mempelajari pola fluktuasi pasar. 

### B. Bagaimana Model Baru Memecahkan Masalah Ini?
Model baru memecahkan masalah ini dengan pendekatan yang jauh lebih menantang bagi AI:
1. **Target Shifting:** Model dipaksa menganalisis data hari ini (RSI, MACD, dll) untuk memprediksi kejadian di hari **esok (H+1)**. 
2. **Fitur Ekstraksi Murni:** Kami meniadakan MA dari daftar fitur latih, sehingga AI dituntut untuk mencari korelasi tersembunyi antara kekuatan momentum (RSI) dan volatilitas harga (Bollinger Bands) yang mengindikasikan akan terjadinya persilangan MA (*Golden Cross*).

### C. Alasan Pemilihan Algoritma Random Forest
Meskipun kami telah merombak arsitekturnya, kami tetap mempertahankan algoritma **Random Forest Classifier**. Alasannya:
- **Ketahanan terhadap Noise:** Harga saham setiap detiknya bergerak fluktuatif (*noise*). Algoritma linear sangat rentan tertipu oleh *noise* ini, sedangkan Random Forest sangat kebal.
- **Analisis Multi-Dimensi:** Mampu memahami hubungan non-linear. Misalnya: *Jika RSI tinggi, TAPI Bollinger Bands sedang sempit, maka apa yang terjadi?* Random Forest unggul dalam mengombinasikan berbagai kondisi ini secara bersamaan tanpa mengalami *overfitting*.

---
*Dokumen ini dapat digunakan sebagai referensi bacaan (speaker notes) saat mendemokan hasil program di depan kelas.*
