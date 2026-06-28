"""
AttritionGuard - Employee Attrition Risk Prediction
Deployment app (Streamlit)
Dibuat berdasarkan pipeline preprocessing & feature engineering
di notebooks/analysis.ipynb (Data Prep: Zahra, ML: Hafizatul)

Cara jalanin lokal:
    streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AttritionGuard - Employee Risk Prediction",
    page_icon="🛡️",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ----------------------------------------------------------------------------
# 2. LOAD MODEL & ARTEFAK (di-cache supaya tidak load ulang setiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    threshold = joblib.load(os.path.join(MODEL_DIR, "optimal_threshold.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return model, feature_columns, threshold, scaler


try:
    model, FEATURE_COLUMNS, OPTIMAL_THRESHOLD, scaler = load_artifacts()
    ARTIFACTS_OK = True
except FileNotFoundError as e:
    ARTIFACTS_OK = False
    LOAD_ERROR = str(e)


# ----------------------------------------------------------------------------
# 3. PREPROCESSING — HARUS SAMA PERSIS DENGAN NOTEBOOK
#    (cleaning -> encoding -> feature engineering -> scaling -> reindex)
# ----------------------------------------------------------------------------
BINARY_ENCODED_EXCLUDE = ["Gender", "OverTime", "Is_Overworked", "Long_No_Promotion"]


def build_feature_row(raw: dict) -> pd.DataFrame:
    """Ubah satu input mentah (dict dari form) jadi 1 baris DataFrame
    yang strukturnya identik dengan X_preprocessed.csv di notebook."""

    df = pd.DataFrame([raw])

    # --- Binary encoding (samakan dengan LabelEncoder alfabetis di notebook) ---
    # Gender: Female=0, Male=1
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    # OverTime: No=0, Yes=1
    df["OverTime"] = df["OverTime"].map({"No": 0, "Yes": 1})

    # --- One-Hot Encoding untuk kolom kategorikal multi-kelas (drop_first=True) ---
    onehot_cols = ["BusinessTravel", "Department", "EducationField",
                   "JobRole", "MaritalStatus"]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=False)

    # --- Feature Engineering (identik dengan notebook) ---
    df["Salary_Experience_Ratio"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    df["Career_Stagnation_Ratio"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
    satisfaction_cols = ["JobSatisfaction", "EnvironmentSatisfaction",
                         "RelationshipSatisfaction", "JobInvolvement"]
    df["Overall_Satisfaction_Index"] = df[satisfaction_cols].mean(axis=1)
    df["Long_No_Promotion"] = (df["YearsSinceLastPromotion"] > 3).astype(int)
    df["Is_Overworked"] = df["OverTime"].astype(int)
    df["Income_JobLevel_Ratio"] = df["MonthlyIncome"] / df["JobLevel"]

    bins = [0, 25, 35, 45, 100]
    labels = ["Gen_Z", "Millennial", "Gen_X", "Boomer"]
    age_group = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    age_dummies = pd.get_dummies(age_group, prefix="Age_Group", drop_first=False)
    df = pd.concat([df, age_dummies], axis=1)

    # --- Samakan dengan kolom hasil training (kolom yang hilang -> 0) ---
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # --- Scaling: hanya kolom numerik non-biner/non-dummy ---
    bool_like = [c for c in df.columns if df[c].dropna().isin([0, 1]).all() and
                 (df[c].dtype != float or set(df[c].unique()).issubset({0, 1}))]
    # ambil ulang scale_cols dengan logika sama seperti notebook:
    scale_cols = [c for c in df.columns
                  if c not in BINARY_ENCODED_EXCLUDE
                  and not c.startswith(("BusinessTravel_", "Department_",
                                        "EducationField_", "JobRole_",
                                        "MaritalStatus_", "Age_Group_"))]
    # scaler dilatih hanya pada kolom2 ini saat training (lihat scaler.feature_names_in_)
    scale_cols = [c for c in scale_cols if c in list(getattr(scaler, "feature_names_in_", scale_cols))]

    df[scale_cols] = scaler.transform(df[scale_cols])

    return df[FEATURE_COLUMNS]


def predict_risk(raw: dict):
    X_row = build_feature_row(raw)
    proba = model.predict_proba(X_row)[:, 1][0]
    label = int(proba >= OPTIMAL_THRESHOLD)
    return proba, label


# ----------------------------------------------------------------------------
# 4. UI
# ----------------------------------------------------------------------------
st.title("🛡️ AttritionGuard")
st.caption("Sistem prediksi risiko resign karyawan — GWE 2026 Data Science Challenge")

if not ARTIFACTS_OK:
    st.error(
        "❌ File model tidak ditemukan di folder `models/`. "
        "Pastikan `best_model.pkl`, `feature_columns.pkl`, "
        "`optimal_threshold.pkl`, dan `scaler.pkl` sudah ada.\n\n"
        f"Detail error: {LOAD_ERROR}"
    )
    st.stop()

tab1, tab2 = st.tabs(["🔍 Prediksi Individu", "ℹ️ Tentang Model"])

with tab1:
    st.subheader("Masukkan Data Karyawan")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demografi & Pekerjaan**")
        age = st.number_input("Usia", 18, 60, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced"])
        department = st.selectbox(
            "Departemen", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Jabatan", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"])
        job_level = st.slider("Job Level", 1, 5, 2)
        business_travel = st.selectbox(
            "Frekuensi Dinas", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        education = st.slider("Tingkat Pendidikan (1-5)", 1, 5, 3)
        education_field = st.selectbox("Bidang Pendidikan", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree",
            "Human Resources", "Other"])

    with col2:
        st.markdown("**Kompensasi & Riwayat Kerja**")
        monthly_income = st.number_input("Gaji Bulanan (MonthlyIncome)", 1000, 25000, 5000)
        daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
        hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)
        monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 14000)
        percent_hike = st.slider("Kenaikan Gaji Terakhir (%)", 11, 25, 14)
        stock_option = st.slider("Stock Option Level", 0, 3, 0)
        num_companies = st.slider("Jumlah Perusahaan Sebelumnya", 0, 9, 1)
        total_working_years = st.slider("Total Tahun Bekerja", 0, 40, 8)
        distance_home = st.slider("Jarak Rumah-Kantor (km)", 1, 30, 5)

    with col3:
        st.markdown("**Kepuasan & Waktu di Perusahaan**")
        overtime = st.selectbox("Lembur (OverTime)", ["No", "Yes"])
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
        job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
        work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
        performance_rating = st.slider("Performance Rating (1-4)", 1, 4, 3)
        training_times = st.slider("Training Times Last Year", 0, 6, 2)
        years_at_company = st.slider("Tahun di Perusahaan", 0, 40, 5)
        years_current_role = st.slider("Tahun di Role Saat Ini", 0, 18, 3)
        years_since_promotion = st.slider("Tahun Sejak Promosi Terakhir", 0, 15, 1)
        years_with_manager = st.slider("Tahun dengan Manajer Saat Ini", 0, 17, 3)

    st.divider()

    if st.button("🔮 Prediksi Risiko Attrition", type="primary", use_container_width=True):
        raw_input = {
            "Age": age, "BusinessTravel": business_travel, "DailyRate": daily_rate,
            "Department": department, "DistanceFromHome": distance_home,
            "Education": education, "EducationField": education_field,
            "EnvironmentSatisfaction": env_satisfaction, "Gender": gender,
            "HourlyRate": hourly_rate, "JobInvolvement": job_involvement,
            "JobLevel": job_level, "JobRole": job_role,
            "JobSatisfaction": job_satisfaction, "MaritalStatus": marital,
            "MonthlyIncome": monthly_income, "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies, "OverTime": overtime,
            "PercentSalaryHike": percent_hike, "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option, "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times, "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company, "YearsInCurrentRole": years_current_role,
            "YearsSinceLastPromotion": years_since_promotion,
            "YearsWithCurrManager": years_with_manager,
        }

        try:
            proba, label = predict_risk(raw_input)

            colA, colB = st.columns([1, 2])
            with colA:
                if label == 1:
                    st.error(f"### ⚠️ RISIKO TINGGI\n**Probabilitas resign: {proba*100:.1f}%**")
                else:
                    st.success(f"### ✅ RISIKO RENDAH\n**Probabilitas resign: {proba*100:.1f}%**")
                st.caption(f"Threshold model: {OPTIMAL_THRESHOLD}")

            with colB:
                st.progress(min(float(proba), 1.0))
                if label == 1:
                    st.markdown(
                        "**Rekomendasi tindakan HR:** lakukan 1-on-1, review kompensasi, "
                        "evaluasi beban kerja & jenjang karier karyawan ini."
                    )
                else:
                    st.markdown("Karyawan ini terindikasi stabil. Tetap pantau secara berkala.")

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
            st.info(
                "Tips debug: cek apakah nama kolom & kategori input di atas sama persis "
                "dengan yang dipakai saat training di notebook (lihat feature_columns.pkl)."
            )

with tab2:
    st.subheader("Tentang Model")
    st.markdown(
        f"""
        - **Dataset**: IBM HR Analytics Employee Attrition & Performance (1.470 karyawan, 35 kolom awal)
        - **Algoritma terbaik**: Logistic Regression (dipilih dari 4 kandidat: Logistic Regression,
          Decision Tree, Random Forest, XGBoost)
        - **Penanganan imbalance**: SMOTE pada data training
        - **Jumlah fitur final**: {len(FEATURE_COLUMNS)} kolom (setelah encoding & feature engineering)
        - **Threshold optimal**: {OPTIMAL_THRESHOLD} (dipilih untuk Recall ≥ 0.70 agar lebih banyak
          kasus resign berhasil terdeteksi, dengan trade-off lebih banyak false alarm)
        - **Fitur tambahan (feature engineering)**: Salary-to-Experience Ratio, Career Stagnation
          Ratio, Overall Satisfaction Index, Long No Promotion Flag, Is Overworked Flag,
          Income-to-JobLevel Ratio, Age Group
        """
    )
    st.caption("Tim ki(C)aw — GWE 2026 Data Science Challenge")
