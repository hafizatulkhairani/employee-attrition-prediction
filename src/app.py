import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AttritionGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Model & Artifacts ───────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model     = joblib.load(os.path.join(base, 'models', 'best_model.pkl'))
    scaler    = joblib.load(os.path.join(base, 'models', 'scaler.pkl'))
    features  = joblib.load(os.path.join(base, 'models', 'feature_columns.pkl'))
    threshold = joblib.load(os.path.join(base, 'models', 'optimal_threshold.pkl'))
    return model, scaler, features, threshold

model, scaler, feature_columns, THRESHOLD = load_artifacts()

# ── Preprocessing Function ───────────────────────────────────
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # 1. Binary Encoding
    df['Gender']  = 1 if data['Gender']  == 'Male' else 0
    df['OverTime'] = 1 if data['OverTime'] == 'Yes'  else 0

    # 2. One-Hot Encoding
    onehot_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    # 3. Feature Engineering
    df['Salary_Experience_Ratio']  = data['MonthlyIncome'] / (data['TotalWorkingYears'] + 1)
    df['Career_Stagnation_Ratio']  = data['YearsInCurrentRole'] / (data['YearsAtCompany'] + 1)
    df['Overall_Satisfaction_Index'] = np.mean([
        data['JobSatisfaction'], data['EnvironmentSatisfaction'],
        data['RelationshipSatisfaction'], data['JobInvolvement']
    ])
    df['Long_No_Promotion'] = 1 if data['YearsSinceLastPromotion'] > 3 else 0
    df['Is_Overworked']     = 1 if data['OverTime'] == 'Yes' else 0
    df['Income_JobLevel_Ratio'] = data['MonthlyIncome'] / data['JobLevel']

    # Age Group
    age = data['Age']
    df['Age_Group_Millennial'] = 1 if 25 <= age < 35 else 0
    df['Age_Group_Gen_X']      = 1 if 35 <= age < 45 else 0
    df['Age_Group_Boomer']     = 1 if age >= 45 else 0

    # 4. Convert bool ke int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 5. Align kolom dengan training data
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    # 6. Scaling
    binary_skip = ['Gender', 'OverTime', 'Is_Overworked', 'Long_No_Promotion']
    bool_skip   = df.select_dtypes(include=['bool', 'uint8']).columns.tolist()
    scale_cols  = [c for c in df.select_dtypes(include=['int64', 'float64']).columns
                   if c not in binary_skip and c not in bool_skip]
    df[scale_cols] = scaler.transform(df[scale_cols])

    return df

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=70)
st.sidebar.title("🛡️ AttritionGuard")
st.sidebar.markdown("*Employee Risk Prediction System*")
st.sidebar.divider()

page = st.sidebar.radio("Navigasi", [
    "🏠 Home",
    "📊 EDA Dashboard",
    "🔮 Prediksi Attrition",
    "📖 About"
])

# ══════════════════════════════════════════════════════════════
# 🏠 HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛡️ AttritionGuard")
    st.subheader("Employee Risk Prediction System")
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Tentang Proyek
        **AttritionGuard** adalah sistem berbasis Machine Learning yang membantu tim HR
        memprediksi risiko karyawan untuk resign (*attrition*) sebelum kejadian terjadi.
        
        Dengan sistem ini, perusahaan dapat melakukan **intervensi dini** seperti:
        - 💰 Review kompensasi & benefit
        - 📈 Rencana pengembangan karir
        - 🤝 Program mentoring & engagement
        - 🏡 Penyesuaian beban kerja

        ### Permasalahan
        Tingkat turnover karyawan yang tinggi berdampak pada:
        - 💸 Biaya rekrutmen & pelatihan yang besar
        - 📉 Penurunan produktivitas tim
        - 🧠 Kehilangan knowledge organisasi

        ### Metodologi
        Pipeline Data Science end-to-end:
        **Data Preparation → EDA → Machine Learning → Deployment**
        """)
    with col2:
        st.metric("Sub-Tema", "Risk Prediction")
        st.metric("Dataset", "IBM HR Analytics")
        st.metric("Model", "Logistic Regression")
        st.metric("AUC-ROC", "0.802")
        st.success("**Tim:** ki(C)aw")

    st.divider()
    st.subheader("🚀 Cara Menggunakan")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**1️⃣ EDA Dashboard**\nEksplorasi data dan visualisasi insight karyawan")
    with c2:
        st.info("**2️⃣ Prediksi Attrition**\nInput data karyawan dan dapatkan prediksi risiko resign")
    with c3:
        st.info("**3️⃣ About**\nInformasi model, metrik, dan tim pengembang")

# ══════════════════════════════════════════════════════════════
# 📊 EDA DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.title("📊 EDA Dashboard")
    st.markdown("Eksplorasi data IBM HR Analytics Employee Attrition")
    st.divider()

    uploaded = st.file_uploader("Upload Dataset CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Dataset berhasil dimuat! Shape: {df.shape}")

        tab1, tab2, tab3 = st.tabs(["📋 Overview", "📊 Distribusi", "🔗 Korelasi"])

        with tab1:
            st.subheader("Preview Data")
            st.dataframe(df.head(10))
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Karyawan", df.shape[0])
            col2.metric("Total Fitur", df.shape[1])
            if 'Attrition' in df.columns:
                resign = (df['Attrition'] == 'Yes').sum()
                col3.metric("Resign", resign)
                col4.metric("Stay", df.shape[0] - resign)

        with tab2:
            if 'Attrition' in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(df, names='Attrition',
                                 title='Distribusi Attrition',
                                 color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if 'Department' in df.columns:
                        dept = df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
                        fig2 = px.bar(dept, x='Department', y='Count', color='Attrition',
                                      title='Attrition per Department',
                                      color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
                        st.plotly_chart(fig2, use_container_width=True)

            num_col = st.selectbox("Pilih fitur numerik untuk histogram:",
                                   df.select_dtypes(include='number').columns.tolist())
            if num_col:
                if 'Attrition' in df.columns:
                    fig3 = px.histogram(df, x=num_col, color='Attrition',
                                        title=f'Distribusi {num_col} by Attrition',
                                        color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
                                        barmode='overlay', opacity=0.7)
                else:
                    fig3 = px.histogram(df, x=num_col, title=f'Distribusi {num_col}')
                st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            num_df = df.select_dtypes(include='number')
            if len(num_df.columns) > 1:
                corr = num_df.corr()
                fig4 = px.imshow(corr, title='Heatmap Korelasi',
                                 color_continuous_scale='RdBu_r', aspect='auto')
                st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("👆 Upload dataset IBM HR Analytics untuk melihat visualisasi")

# ══════════════════════════════════════════════════════════════
# 🔮 PREDIKSI ATTRITION
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Prediksi Attrition":
    st.title("🔮 Prediksi Risiko Attrition Karyawan")
    st.markdown("Isi data karyawan di bawah ini untuk mendapatkan prediksi risiko resign.")
    st.divider()

    with st.form("prediction_form"):
        st.subheader("👤 Data Personal")
        col1, col2, col3 = st.columns(3)
        with col1:
            age            = st.number_input("Age", 18, 65, 30)
            gender         = st.selectbox("Gender", ["Female", "Male"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        with col2:
            education       = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                           help="1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor")
            education_field = st.selectbox("Education Field", [
                "Life Sciences", "Medical", "Marketing",
                "Technical Degree", "Human Resources", "Other"])
            distance_home  = st.number_input("Distance From Home (km)", 1, 100, 10)
        with col3:
            num_companies  = st.number_input("Num Companies Worked", 0, 10, 1)
            total_working  = st.number_input("Total Working Years", 0, 40, 5)
            training_times = st.number_input("Training Times Last Year", 0, 6, 2)

        st.subheader("💼 Data Pekerjaan")
        col1, col2, col3 = st.columns(3)
        with col1:
            department      = st.selectbox("Department", [
                "Sales", "Research & Development", "Human Resources"])
            job_role        = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative",
                "Manager", "Sales Representative", "Research Director", "Human Resources"])
            job_level       = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        with col2:
            business_travel = st.selectbox("Business Travel", [
                "Non-Travel", "Travel_Rarely", "Travel_Frequently"])
            overtime        = st.selectbox("Over Time", ["No", "Yes"])
            stock_option    = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        with col3:
            years_company   = st.number_input("Years At Company", 0, 40, 3)
            years_role      = st.number_input("Years In Current Role", 0, 18, 2)
            years_manager   = st.number_input("Years With Current Manager", 0, 17, 2)
            years_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)

        st.subheader("💰 Data Kompensasi & Kepuasan")
        col1, col2, col3 = st.columns(3)
        with col1:
            monthly_income  = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
            daily_rate      = st.number_input("Daily Rate", 100, 1500, 800)
            hourly_rate     = st.number_input("Hourly Rate", 30, 100, 65)
            monthly_rate    = st.number_input("Monthly Rate", 2000, 27000, 14000)
            pct_salary_hike = st.number_input("Percent Salary Hike (%)", 11, 25, 14)
        with col2:
            job_satisfaction = st.selectbox("Job Satisfaction",         [1, 2, 3, 4],
                                            help="1=Low, 2=Medium, 3=High, 4=Very High")
            env_satisfaction = st.selectbox("Environment Satisfaction",  [1, 2, 3, 4],
                                            help="1=Low, 2=Medium, 3=High, 4=Very High")
            rel_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4],
                                            help="1=Low, 2=Medium, 3=High, 4=Very High")
        with col3:
            job_involvement   = st.selectbox("Job Involvement",   [1, 2, 3, 4],
                                             help="1=Low, 2=Medium, 3=High, 4=Very High")
            work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4],
                                             help="1=Bad, 2=Good, 3=Better, 4=Best")
            perf_rating       = st.selectbox("Performance Rating", [1, 2, 3, 4],
                                             help="1=Low, 2=Good, 3=Excellent, 4=Outstanding")

        submitted = st.form_submit_button("🔮 Prediksi Sekarang", type="primary", use_container_width=True)

    if submitted:
        input_data = {
            'Age'                    : age,
            'BusinessTravel'         : business_travel,
            'DailyRate'              : daily_rate,
            'Department'             : department,
            'DistanceFromHome'       : distance_home,
            'Education'              : education,
            'EducationField'         : education_field,
            'EnvironmentSatisfaction': env_satisfaction,
            'Gender'                 : gender,
            'HourlyRate'             : hourly_rate,
            'JobInvolvement'         : job_involvement,
            'JobLevel'               : job_level,
            'JobRole'                : job_role,
            'JobSatisfaction'        : job_satisfaction,
            'MaritalStatus'          : marital_status,
            'MonthlyIncome'          : monthly_income,
            'MonthlyRate'            : monthly_rate,
            'NumCompaniesWorked'     : num_companies,
            'OverTime'               : overtime,
            'PercentSalaryHike'      : pct_salary_hike,
            'PerformanceRating'      : perf_rating,
            'RelationshipSatisfaction': rel_satisfaction,
            'StockOptionLevel'       : stock_option,
            'TotalWorkingYears'      : total_working,
            'TrainingTimesLastYear'  : training_times,
            'WorkLifeBalance'        : work_life_balance,
            'YearsAtCompany'         : years_company,
            'YearsInCurrentRole'     : years_role,
            'YearsSinceLastPromotion': years_promotion,
            'YearsWithCurrManager'   : years_manager,
        }

        try:
            X_input = preprocess_input(input_data)
            prob    = model.predict_proba(X_input)[0][1]
            pred    = 1 if prob >= THRESHOLD else 0

            st.divider()
            st.subheader("📊 Hasil Prediksi")

            col1, col2, col3 = st.columns(3)
            with col1:
                if pred == 1:
                    st.error("⚠️ **BERISIKO RESIGN**")
                else:
                    st.success("✅ **TIDAK BERISIKO**")
            with col2:
                st.metric("Probabilitas Resign", f"{prob*100:.1f}%")
            with col3:
                st.metric("Threshold", f"{THRESHOLD*100:.0f}%")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = prob * 100,
                title = {'text': "Risiko Attrition (%)"},
                delta = {'reference': THRESHOLD * 100},
                gauge = {
                    'axis' : {'range': [0, 100]},
                    'bar'  : {'color': '#e74c3c' if pred == 1 else '#2ecc71'},
                    'steps': [
                        {'range': [0,  40],  'color': '#d5f5e3'},
                        {'range': [40, 70],  'color': '#fdebd0'},
                        {'range': [70, 100], 'color': '#fadbd8'},
                    ],
                    'threshold': {
                        'line' : {'color': 'black', 'width': 4},
                        'thickness': 0.75,
                        'value': THRESHOLD * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Rekomendasi
            st.subheader("💡 Rekomendasi HR")
            if pred == 1:
                st.warning(f"""
                **Karyawan ini berisiko resign (probabilitas: {prob*100:.1f}%)**
                
                Tindakan yang disarankan:
                - 💰 Lakukan review kompensasi dan benefit
                - 📈 Diskusikan rencana pengembangan karir
                - 🤝 Tingkatkan engagement dan keterlibatan
                - 🏡 Evaluasi beban kerja dan work-life balance
                - 👨‍💼 Jadwalkan 1-on-1 dengan manager
                """)
            else:
                st.success(f"""
                **Karyawan ini tidak berisiko resign (probabilitas: {prob*100:.1f}%)**
                
                Tetap pertahankan kondisi positif dengan:
                - ✅ Pertahankan lingkungan kerja yang baik
                - ✅ Berikan apresiasi dan recognition
                - ✅ Lanjutkan program pengembangan yang ada
                """)

            # Faktor risiko
            st.subheader("🔍 Faktor Risiko Utama")
            risk_factors = []
            if overtime == "Yes":
                risk_factors.append(("⚠️ Overtime", "Karyawan sering lembur"))
            if business_travel == "Travel_Frequently":
                risk_factors.append(("⚠️ Travel Frequently", "Sering perjalanan bisnis"))
            if job_satisfaction <= 2:
                risk_factors.append(("⚠️ Job Satisfaction Rendah", f"Nilai: {job_satisfaction}/4"))
            if env_satisfaction <= 2:
                risk_factors.append(("⚠️ Environment Satisfaction Rendah", f"Nilai: {env_satisfaction}/4"))
            if years_promotion > 3:
                risk_factors.append(("⚠️ Lama Tidak Dipromosi", f"{years_promotion} tahun"))
            if work_life_balance <= 2:
                risk_factors.append(("⚠️ Work-Life Balance Buruk", f"Nilai: {work_life_balance}/4"))
            if marital_status == "Single":
                risk_factors.append(("ℹ️ Status Single", "Lebih mobile, lebih mudah pindah"))

            if risk_factors:
                for factor, desc in risk_factors:
                    st.warning(f"**{factor}** — {desc}")
            else:
                st.info("✅ Tidak ada faktor risiko signifikan yang terdeteksi")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("Pastikan semua file model tersedia di folder `models/`")

# ══════════════════════════════════════════════════════════════
# 📖 ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "📖 About":
    st.title("📖 About AttritionGuard")
    st.divider()

    st.subheader("👥 Tim ki(C)aw")
    team_data = {
        "Nama"  : ["Hafizatul Khairani", "Zahra Daniah", "Nailah Fauziyyah", "Nabila Nur Aini"],
        "Tugas" : ["Machine Learning", "Data Preparation", "EDA", "Deployment"]
    }
    team_df = pd.DataFrame(team_data)
    team_df.index = team_df.index + 1
    st.table(pd.DataFrame(team_data))

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Model")
        st.markdown("""
        - **Algoritma:** Logistic Regression
        - **Handling Imbalance:** SMOTE
        - **Cross Validation:** StratifiedKFold (5-fold)
        - **Threshold Optimal:** 0.2 (prioritas Recall)
        """)

        st.subheader("📊 Performa Model")
        metrics = {
            "Metrik"  : ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
            "Score"   : ["84.35%", "51.35%", "40.43%", "45.24%", "80.22%"]
        }
        metrics_df = pd.DataFrame(metrics)
        metrics_df.index = metrics_df.index + 1
        st.table(pd.DataFrame(metrics))

    with col2:
        st.subheader("🔧 Pipeline")
        st.markdown("""
        1. **Data Preparation** (Zahra)
           - Cleaning, Preprocessing, Feature Engineering
        2. **EDA** (Nailah)
           - Analisis & Visualisasi
        3. **Machine Learning** (Hafizatul)
           - Training, Evaluasi, Threshold Optimization
        4. **Deployment** (Nabila)
           - Streamlit App
        """)

        st.subheader("📊 Dataset")
        st.markdown("""
        **IBM HR Analytics Employee Attrition**  
        [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
        - 1.470 baris, 35 kolom
        - 16.1% karyawan Attrition (Yes)
        """)

    st.divider()
    st.subheader("🤖 Penggunaan AI Tools")
    st.markdown("""
    - **Claude (Anthropic)** — Bantuan penulisan kode dan dokumentasi
    - **ChatGPT (OpenAI)** — Bantuan brainstorming dan debugging
    """)
    st.caption("GWE 2026 Data Science Challenge | Grow With EDM Gen 7")