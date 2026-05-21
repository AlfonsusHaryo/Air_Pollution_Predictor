# AQI Streamlit Deployment — Step by Step

Project ini adalah template deployment untuk model klasifikasi `AQI Category` dari notebook final tanpa hyperparameter tuning.

## Rekomendasi Model

Untuk deployment, gunakan **1 model final saja**.

Alasannya:
- Aplikasi deployment harus fokus pada prediksi akhir, bukan eksperimen perbandingan.
- Model lain cukup dipakai di notebook sebagai proses seleksi.
- Aplikasi jadi lebih mudah dijelaskan saat presentasi.
- Risiko kebingungan user lebih kecil karena hanya ada satu output model.

Jadi alurnya:
**Notebook = training, evaluasi, seleksi model**  
**Deployment = memakai 1 model final terbaik**

## Struktur Folder

```text
aqi_streamlit_deployment/
├── app.py
├── export_model_from_csv.py
├── requirements.txt
├── README_STEP_BY_STEP.md
├── sample_input.csv
├── .streamlit/
│   └── config.toml
└── model/
    ├── best_model_no_tuning.pkl
    ├── scaler_no_tuning.pkl
    └── model_metadata_no_tuning.json
```

## File yang Wajib Ada

Aplikasi membutuhkan 3 file model:

```text
model/best_model_no_tuning.pkl
model/scaler_no_tuning.pkl
model/model_metadata_no_tuning.json
```

File tersebut bisa didapat dari notebook final bagian **Export Model dan Metadata**.

Jika notebook kamu menghasilkan folder `outputs_no_tuning/`, ambil file ini:

```text
outputs_no_tuning/best_model_no_tuning.pkl
outputs_no_tuning/scaler_no_tuning.pkl
outputs_no_tuning/model_metadata_no_tuning.json
```

Lalu copy ke folder:

```text
aqi_streamlit_deployment/model/
```

## Cara Menjalankan di Laptop

### 1. Buka folder project

```bash
cd aqi_streamlit_deployment
```

### 2. Buat virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install library

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

Nanti aplikasi akan terbuka di browser.

## Cara Membuat File Model Jika Belum Ada

Kalau kamu belum punya file `.pkl` dan `.json`, gunakan cara ini:

1. Letakkan dataset `global_air_pollution_dataset.csv` di folder yang sama dengan `export_model_from_csv.py`.
2. Jalankan:

```bash
python export_model_from_csv.py
```

3. Setelah selesai, folder `model/` akan berisi file yang dibutuhkan aplikasi.

## Cara Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru.
2. Upload semua isi folder `aqi_streamlit_deployment`.
3. Pastikan file model ada di folder `model/`.
4. Buka Streamlit Community Cloud.
5. Pilih repository.
6. Main file path isi:

```text
app.py
```

7. Klik Deploy.

## Penjelasan Saat Presentasi

Kamu bisa jelaskan seperti ini:

> Pada tahap notebook, kelompok kami melakukan preprocessing, EDA, training, evaluasi, dan seleksi model. Setelah model terbaik dipilih, deployment hanya menggunakan satu model final agar aplikasi lebih stabil, sederhana, dan mudah digunakan. Model lain tidak ditampilkan pada aplikasi karena hanya berfungsi sebagai pembanding saat proses eksperimen.

## Kenapa Tidak Pakai 2 Model di Aplikasi?

Tidak disarankan, kecuali tugasnya memang membuat comparison dashboard.

Untuk tugas klasifikasi, 2 model di aplikasi bisa membuat fokus menjadi melebar. Dosen bisa menganggap aplikasi kamu bukan sistem prediksi final, tetapi masih eksperimen. Karena itu, aplikasi ini memakai **1 model final terbaik**.

## Fitur Aplikasi

- Prediksi manual dengan slider.
- Tampilan hasil prediksi yang modern.
- Confidence score jika model mendukung `predict_proba`.
- Grafik probabilitas kategori.
- Grafik nilai polutan.
- Batch prediction via CSV.
- Download hasil prediksi CSV.
- Ringkasan model.
- Penjelasan fitur.
- Panduan penggunaan.

## Troubleshooting

### Error: File model belum ditemukan

Pastikan 3 file ini ada:

```text
model/best_model_no_tuning.pkl
model/scaler_no_tuning.pkl
model/model_metadata_no_tuning.json
```

### Error: ModuleNotFoundError

Jalankan:

```bash
pip install -r requirements.txt
```

### Error saat upload CSV

Pastikan nama kolom CSV sama persis:

```text
CO AQI Value
Ozone AQI Value
NO2 AQI Value
PM2.5 AQI Value
```

## Catatan Akademik

Aplikasi ini tidak memakai `AQI Value` sebagai input karena kolom tersebut berpotensi menyebabkan data leakage. Input deployment hanya menggunakan empat fitur polutan utama yang juga dipakai pada model final.
