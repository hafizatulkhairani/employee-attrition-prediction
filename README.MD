# 🛡️ AttritionGuard - Employee Risk Prediction System

> **GWE 2026 Data Science Challenge** | Sub-Tema: Risk Prediction  
> Tema Utama: *Leveraging Data for Societal and Industrial Impact*

---

## 👥 Tim ki(C)aw

| Nama | Tugas |
|------|-------|
| Hafizatul Khairani | Machine Learning |
| Zahra Daniah | Data Preparation |
| Nailah Fauziyyah | Exploratory Data Analysis (EDA) |
| Nabila Nur Aini | Deployment |

---

## 📋 Pembagian Tugas Detail

### 🗂️ Zahra Daniah — Data Preparation
- Data Cleaning
- Data Preprocessing
- Feature Engineering

### 📊 Nailah Fauziyyah — Exploratory Data Analysis (EDA)
- Analisis karakteristik data
- Visualisasi data
- Insight dan storytelling

### 🤖 Hafizatul Khairani — Machine Learning
- Pemilihan model
- Training model
- Evaluasi model
- Hyperparameter Tuning

### 🚀 Nabila Nur Aini — Deployment
- Deployment menggunakan Streamlit
- Memastikan aplikasi dapat diakses secara publik
- Memastikan fitur analisis dan prediksi berjalan dengan baik

---

## 📌 Deskripsi Proyek

**AttritionGuard** adalah sistem prediksi risiko attrition (pengunduran diri) karyawan berbasis Machine Learning. Sistem ini membantu perusahaan mengidentifikasi karyawan yang berpotensi resign sebelum kejadian terjadi, sehingga perusahaan dapat mengambil tindakan preventif untuk mempertahankan talenta terbaik mereka.

### Permasalahan
Tingkat turnover karyawan yang tinggi menjadi tantangan serius bagi perusahaan karena berdampak langsung pada produktivitas dan biaya operasional. Dengan pendekatan data-driven, perusahaan dapat mendeteksi risiko attrition lebih awal.

### Solusi
Membangun model klasifikasi Machine Learning yang mampu memprediksi apakah seorang karyawan berisiko untuk resign berdasarkan berbagai faktor seperti kepuasan kerja, beban kerja, kompensasi, dan lain-lain.

---

## 📊 Dataset

- **Sumber:** [Employee Attrition Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset)
- **Deskripsi:** Dataset berisi informasi karyawan beserta label attrition (resign/tidak)

---

## 🔧 Tech Stack

- **Language:** Python 3.10+
- **ML Libraries:** Scikit-learn, XGBoost, SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit

---

## 📁 Struktur Repository
attritionguard/

├── README.md

├── requirements.txt

├── notebooks/

│   └── analysis.ipynb

├── src/

│   └── app.py

├── data/

│   └── employee_attrition_dataset_10000.csv

├── models/

└── presentation/

└── slides.pdf


---

## 🚀 Cara Menjalankan Aplikasi

### 1. Clone Repository
```bash
git clone https://github.com/hafizatulkhairani/attritionguard.git
cd attritionguard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit
```bash
streamlit run src/app.py
```

---

## 🔄 Cara Kontribusi (untuk Anggota Tim)

### Pertama kali (clone dulu)
```bash
git clone https://github.com/hafizatulkhairani/attritionguard.git
cd attritionguard
```

### Setiap kali mau push kode
```bash
git add .
git commit -m "deskripsi singkat apa yang dikerjain"
git push
```

### Kalau ada update dari temen yang lain, pull dulu sebelum kerja
```bash
git pull
```

---

## 🌐 Link Deployment

> 🔗 Coming soon...

---

## 🤖 Penggunaan AI Tools

- **Claude (Anthropic)** - Bantuan penulisan kode dan dokumentasi
- **ChatGPT (OpenAI)** - Bantuan brainstorming dan debugging

---

*GWE 2026 Data Science Challenge | Grow With EDM Gen 7*