# Penentuan Nilai Properti Berdasarkan Tren Pasar dengan Time Series Analysis
## Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis dan memprediksi nilai properti berdasarkan tren pasar menggunakan teknik Time Series Analysis. Dalam proyek ini, data historis harga properti, tingkat inflasi, dan suku bunga dari tahun 2010 hingga 2025 digunakan untuk membangun model yang dapat memprediksi nilai properti di masa depan. Dengan memahami tren pasar, proyek ini memberikan wawasan berharga bagi pengembang properti, investor, dan pemilik properti untuk pengambilan keputusan strategis.

## Latar Belakang

Perubahan nilai properti sangat dipengaruhi oleh berbagai faktor seperti kondisi ekonomi, tingkat inflasi, suku bunga, dan permintaan pasar. Dengan analisis deret waktu, kita dapat mengidentifikasi pola historis dan memprediksi harga di masa depan. Proyek ini menggunakan metode seperti ARIMA, SARIMA, atau LSTM untuk menghasilkan prediksi yang akurat.

## Dataset

Sumber Data: Dataset historis mencakup informasi berikut:

- Harga Properti: Nilai properti berdasarkan wilayah.

- Inflasi: Tingkat inflasi tahunan.

- Suku Bunga: Tingkat suku bunga yang memengaruhi pembiayaan properti.

- Periode: 2010 – 2025

## Alur Proyek

1. Pengumpulan Data:

    - Mengimpor data dari sumber yang valid.

2. Preprocessing Data:

    - Penanganan nilai kosong, normalisasi, dan transformasi data.

3. Analisis Data Eksploratif (EDA):

    - Mengidentifikasi tren, pola musiman, dan anomali.

4. Feature Engineering:

    - Membuat fitur baru seperti rata-rata bergerak dan perbedaan harga.

5. Modeling:

    - Menggunakan algoritma deret waktu untuk prediksi.

6. Evaluasi:

    - Menggunakan metrik seperti MAE, RMSE, dan MAPE untuk menilai performa model.

## Teknologi yang Digunakan

- Bahasa Pemrograman: Python

- Library Utama:

    - Pandas

    - NumPy

    - Statsmodels

    - Scikit-learn

    - Matplotlib/Seaborn

- Jupyter Notebook untuk dokumentasi dan eksplorasi interaktif.

## Tujuan

- Membuat model prediksi harga properti berbasis tren pasar.

- Memberikan wawasan kepada pengguna mengenai faktor yang memengaruhi perubahan harga.

- Meningkatkan akurasi prediksi untuk mendukung pengambilan keputusan.

## Hasil dan Kesimpulan

Hasil:
 - Model menunjukkan performa yang baik dengan garis diagonal merah, berarti model memiliki prediksi yang baik.
 - Sebaran yang jauh menunjukkan kelemahan dalam beberapa prediksi, kemungkinan karena data outlier atau fitur yang kurang relevan.

Kesimpulan:
 - Model Random Forest memberikan hasil yang cukup baik. Ini menunjukkan bahwa sebagian besar variasi harga   properti dapat dijelaskan oleh fitur yang digunakan.
 - LANDAREA memiliki pengaruh paling besar terhadap prediksi harga properti, diikuti oleh fitur lain seperti jumlah lantai (STORIES).
 - Menggunakan metode lain seperti Gradient Boosting atau Time Series Analysis jika data memiliki sifat musiman atau tren.

## Cara Menjalankan Proyek

- Clone repositori ini:
```
git clone https://github.com/username/property-value-prediction.git
```
- Install dependensi:
```
pip install -r requirements.txt
```
- Jalankan notebook Jupyter:
```
jupyter notebook
```
- Buka file Time_Series_Property_Analysis.ipynb untuk melihat langkah-langkah analisis dan prediksi.
