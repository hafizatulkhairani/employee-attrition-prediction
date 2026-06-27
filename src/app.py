import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="AttritionGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ──────────────────────────────────────────
st.sidebar.title("🛡️ AttritionGuard")
st.sidebar.markdown("*Employee Risk Prediction System*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "📊 EDA Dashboard", "🔮 Prediction", "📖 About"]
)

# ── Home ─────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🛡️ AttritionGuard")
    st.subheader("Employee Risk Prediction System")
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Tentang Proyek
        **AttritionGuard** adalah sistem berbasis Machine Learning yang membantu perusahaan 
        memprediksi risiko karyawan untuk resign (*attrition*).

        ### Permasalahan
        Tingkat turnover karyawan yang tinggi berdampak pada:
        - 💸 Biaya rekrutmen & pelatihan yang besar
        - 📉 Penurunan produktivitas tim
        - 🧠 Kehilangan knowledge organisasi

        ### Solusi
        Sistem ini menganalisis faktor-faktor yang mempengaruhi keputusan karyawan 
        untuk resign dan memberikan prediksi risiko secara real-time.
        """)
    with col2:
        st.info("**Sub-Tema:** Risk Prediction")
        st.info("**Dataset:** Employee Attrition (Kaggle)")
        st.success("**Tim:** ki(C)aw")

# ── EDA Dashboard ─────────────────────────────────────
elif page == "📊 EDA Dashboard":
    st.title("📊 EDA Dashboard")
    st.info("Upload dataset untuk melihat visualisasi eksplorasi data.")

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Dataset berhasil dimuat! Shape: {df.shape}")
        st.subheader("Preview Data")
        st.dataframe(df.head(10))

        if "Attrition" in df.columns:
            st.subheader("Distribusi Attrition")
            fig = px.pie(df, names="Attrition", title="Distribusi Attrition")
            st.plotly_chart(fig, use_container_width=True)

# ── Prediction ────────────────────────────────────────
elif page == "🔮 Prediction":
    st.title("🔮 Employee Attrition Prediction")
    st.markdown("Isi informasi karyawan untuk mendapatkan prediksi risiko attrition.")
    st.info("🚧 Model sedang dalam pengembangan.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Usia", 18, 60, 30)
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        overtime = st.selectbox("Overtime", ["Yes", "No"])
    with col2:
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])

    if st.button("🔮 Prediksi Risiko", type="primary"):
        st.warning("⚠️ Model belum tersedia. Akan diupdate setelah training selesai.")

# ── About ─────────────────────────────────────────────
elif page == "📖 About":
    st.title("📖 About")

    st.subheader("👥 Tim ki(C)aw")
    team = {"Nama": ["Hafizatul Khairani", "Zahra Daniah", "Nailah Fauziyyah", "Nabila Nur Aini"]}
    st.table(pd.DataFrame(team))

    st.subheader("🔧 Model & Metodologi")
    st.markdown("""
    - **Algoritma:** Coming soon setelah training
    - **Metrik Evaluasi:** Accuracy, Precision, Recall, F1-Score, AUC-ROC
    - **Pipeline:** Data Cleaning → EDA → Feature Engineering → Modeling → Deployment
    """)

    st.subheader("📊 Dataset")
    st.markdown("[Employee Attrition Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset)")