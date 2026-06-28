"""
AttritionGuard — Sistem Prediksi Risiko Attrition Karyawan
============================================================
Dikerjakan oleh: Nabila Nur Aini (Deployment)
Model & artefak dihasilkan oleh tahap Machine Learning (Hafizatul Khairani)
Pipeline preprocessing mereplikasi notebook Data Preparation (Zahra Daniah)

Jalankan dengan:
    streamlit run app.py
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Konfigurasi halaman
st.set_page_config(
    page_title="AttritionGuard - Prediksi Risiko Attrition",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1.2 Lokasi File & Loader Artefak Model
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

TARGET_COL = "Attrition"
# Kolom yang dibuang saat cleaning (notebook cell 3.3)
DROP_COLS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]


def _find_file(candidates, folders):
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in candidates:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return p
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    art = {"model": None, "scaler": None, "feature_columns": None,
           "threshold": 0.5, "errors": []}

    model_path = _find_file(["best_model.pkl", "model.pkl"], [MODELS_DIR, BASE_DIR, DATA_DIR])
    scaler_path = _find_file(["scaler.pkl"], [MODELS_DIR, BASE_DIR, DATA_DIR])
    feat_path = _find_file(["feature_columns.pkl"], [MODELS_DIR, BASE_DIR, DATA_DIR])
    thr_path = _find_file(["optimal_threshold.pkl"], [MODELS_DIR, BASE_DIR, DATA_DIR])

    try:
        if model_path:
            art["model"] = joblib.load(model_path)
        else:
            art["errors"].append("best_model.pkl tidak ditemukan di folder models/")
    except Exception as e:
        art["errors"].append(f"Gagal memuat model: {e}")

    try:
        if scaler_path:
            art["scaler"] = joblib.load(scaler_path)
        else:
            art["errors"].append("scaler.pkl tidak ditemukan")
    except Exception as e:
        art["errors"].append(f"Gagal memuat scaler: {e}")

    try:
        if feat_path:
            art["feature_columns"] = list(joblib.load(feat_path))
    except Exception as e:
        art["errors"].append(f"Gagal memuat feature_columns: {e}")

    try:
        if thr_path:
            art["threshold"] = float(joblib.load(thr_path))
    except Exception:
        pass

    # Fallback feature_columns dari atribut model bila tersedia
    if art["feature_columns"] is None and art["model"] is not None:
        cols = getattr(art["model"], "feature_names_in_", None)
        if cols is not None:
            art["feature_columns"] = list(cols)

    return art


# ============================================================
# 1.3 Loader Dataset Referensi & Schema Encoding
# ============================================================
def _looks_raw(df):
    """True jika dataframe masih mentah (punya kolom kategorikal asli, belum di-encode)."""
    if df is None:
        return False
    cols = set(df.columns)
    raw_markers = {"JobRole", "Department", "BusinessTravel", "MaritalStatus"}
    encoded_markers = {"Salary_Experience_Ratio", "Age_Group_Millennial",
                       "MaritalStatus_Single", "Is_Overworked"}
    return len(raw_markers & cols) >= 2 and len(encoded_markers & cols) == 0


@st.cache_data(show_spinner=False)
def load_reference_data():
    """Dataset mentah untuk membangun form input & dashboard EDA.
    Selalu mengutamakan CSV mentah (kolom kategorikal asli), bukan yang sudah di-encode."""
    preferred = [
        "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "HR-Employee-Attrition.csv",
        "employee_attrition_dataset_10000.csv",
        "employee_attrition_clean.csv",
    ]
    paths = []
    for name in preferred:
        p = _find_file([name], [DATA_DIR, BASE_DIR])
        if p:
            paths.append(p)
    if os.path.isdir(DATA_DIR):
        for n in sorted(os.listdir(DATA_DIR)):
            if n.lower().endswith(".csv"):
                p = os.path.join(DATA_DIR, n)
                if p not in paths:
                    paths.append(p)
    fallback = None
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if fallback is None:
            fallback = df
        if _looks_raw(df):
            return df
    return fallback


@st.cache_data(show_spinner=False)
def build_schema(_ref_df):
    """Tentukan kolom binary vs one-hot + mapping encoding dari dataset mentah."""
    df = _ref_df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != TARGET_COL]
    binary_cols, onehot_cols, binary_maps = [], [], {}
    for c in cat_cols:
        uniques = sorted([str(x) for x in df[c].dropna().unique()])
        if len(uniques) == 2:
            binary_cols.append(c)
            binary_maps[c] = {v: i for i, v in enumerate(uniques)}  # LabelEncoder = alfabetis
        else:
            onehot_cols.append(c)
    num_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns
                if c != TARGET_COL]
    return {
        "binary_cols": binary_cols,
        "onehot_cols": onehot_cols,
        "binary_maps": binary_maps,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "raw_columns": [c for c in df.columns if c != TARGET_COL],
    }


# ============================================================
# 2.1 Praproses — Replikasi Pipeline Notebook Data Preparation
# ============================================================
def engineer_features(df):
    df = df.copy()
    if {"MonthlyIncome", "TotalWorkingYears"}.issubset(df.columns):
        df["Salary_Experience_Ratio"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    if {"YearsInCurrentRole", "YearsAtCompany"}.issubset(df.columns):
        df["Career_Stagnation_Ratio"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
    sat = [c for c in ["JobSatisfaction", "EnvironmentSatisfaction",
                       "RelationshipSatisfaction", "JobInvolvement"] if c in df.columns]
    if sat:
        df["Overall_Satisfaction_Index"] = df[sat].mean(axis=1)
    if "YearsSinceLastPromotion" in df.columns:
        df["Long_No_Promotion"] = (df["YearsSinceLastPromotion"] > 3).astype(int)
    if "OverTime" in df.columns:
        df["Is_Overworked"] = df["OverTime"].astype(int)
    if {"MonthlyIncome", "JobLevel"}.issubset(df.columns):
        df["Income_JobLevel_Ratio"] = df["MonthlyIncome"] / df["JobLevel"]
    if "Age" in df.columns:
        bins = [0, 25, 35, 45, 100]
        labels = ["Gen_Z", "Millennial", "Gen_X", "Boomer"]
        df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
        df = pd.get_dummies(df, columns=["Age_Group"], drop_first=True)
    return df


def preprocess(raw_df, schema, feature_columns, scaler):
    """Ubah data mentah -> matriks fitur siap prediksi (sesuai model)."""
    df = raw_df.copy()
    df = df.drop(columns=[c for c in DROP_COLS + [TARGET_COL] if c in df.columns],
                 errors="ignore")

    # 1. Binary encoding (alfabetis, seperti LabelEncoder)
    for c in schema["binary_cols"]:
        if c in df.columns:
            df[c] = df[c].astype(str).map(schema["binary_maps"][c]).fillna(0).astype(int)

    # 2. One-hot encoding multi-kategori
    onehot = [c for c in schema["onehot_cols"] if c in df.columns]
    if onehot:
        df = pd.get_dummies(df, columns=onehot, drop_first=True)

    # 3. Feature engineering (termasuk Age_Group one-hot)
    df = engineer_features(df)

    # 4. Selaraskan ke kolom yang diharapkan model
    if feature_columns is not None:
        df = df.reindex(columns=feature_columns, fill_value=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

    # 5. Scaling hanya pada kolom yang dipakai saat fit scaler
    if scaler is not None:
        sc_cols = list(getattr(scaler, "feature_names_in_", []))
        sc_cols = [c for c in sc_cols if c in df.columns]
        if sc_cols:
            df[sc_cols] = scaler.transform(df[sc_cols])
    return df


# ============================================================
# 2.2 Fungsi Prediksi & Kategori Risiko
# ============================================================
def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def risk_band(prob):
    if prob < 0.30:
        return "Rendah", "#2ecc71"
    if prob < 0.60:
        return "Sedang", "#f39c12"
    return "Tinggi", "#e74c3c"


# ============================================================
# 2.3 Rekomendasi Tindakan (Rule-Based dari Insight EDA)
# ============================================================
def build_recommendations(row, ref_df):
    recs = []
    def med(col):
        try:
            return float(ref_df[col].median()) if ref_df is not None and col in ref_df else None
        except Exception:
            return None

    if str(row.get("OverTime", "")).lower() in ("yes", "1", "true"):
        recs.append(("Beban kerja / OverTime",
                     "Kurangi lembur, evaluasi distribusi beban kerja, dan pertimbangkan penambahan staf."))
    jl = row.get("JobLevel")
    if jl is not None and float(jl) <= 1:
        recs.append(("Karyawan entry-level",
                     "Berikan mentoring, jalur karier yang jelas, dan paket kompensasi kompetitif."))
    mi = row.get("MonthlyIncome")
    m_med = med("MonthlyIncome")
    if mi is not None and m_med and float(mi) < 0.85 * m_med:
        recs.append(("Kompensasi di bawah median",
                     "Tinjau ulang struktur gaji; gap kompensasi adalah pemicu utama turnover."))
    if str(row.get("MaritalStatus", "")).lower() == "single":
        recs.append(("Status single / karyawan muda",
                     "Tingkatkan employee engagement, benefit, dan keseimbangan kerja-hidup."))
    ysp = row.get("YearsSinceLastPromotion")
    if ysp is not None and float(ysp) > 3:
        recs.append(("Stagnasi karier",
                     "Buat rencana pengembangan & peluang promosi; sudah >3 tahun tanpa promosi."))
    bt = str(row.get("BusinessTravel", "")).lower()
    if "frequent" in bt:
        recs.append(("Sering perjalanan dinas",
                     "Atur ulang jadwal perjalanan, berikan kompensasi/istirahat yang memadai."))
    sat_cols = [c for c in ["JobSatisfaction", "EnvironmentSatisfaction",
                            "RelationshipSatisfaction", "JobInvolvement"] if c in row]
    if sat_cols:
        avg_sat = np.mean([float(row[c]) for c in sat_cols])
        if avg_sat < 2.5:
            recs.append(("Kepuasan kerja rendah",
                         "Lakukan stay interview, perbaiki lingkungan kerja & hubungan tim."))
    if not recs:
        recs.append(("Pertahankan kondisi baik",
                     "Indikator risiko rendah. Lanjutkan praktik retensi & pantau kepuasan berkala."))
    return recs


# ============================================================
# 2.4 Komponen UI — Gauge Chart & Penjelasan Kontribusi Fitur
# ============================================================
def gauge_chart(prob):
    band, color = risk_band(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": f"Probabilitas Resign — Risiko {band}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#eafaf1"},
                {"range": [30, 60], "color": "#fef5e7"},
                {"range": [60, 100], "color": "#fdedec"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def shap_explanation(model, X_row, feature_columns, X_background=None, top_n=10):
    """Hitung kontribusi fitur untuk 1 prediksi. Return (DataFrame[Fitur, Kontribusi], mode).

    Mendukung:
    - Model tree-based (RandomForest, DecisionTree, XGBoost) -> shap.TreeExplainer
    - Logistic Regression / model linear -> shap.LinearExplainer, fallback ke kontribusi
      manual (koefisien * nilai fitur ternormalisasi), karena TreeExplainer tidak berlaku
      untuk model linear.
    """
    # --- Model linear (Logistic Regression dkk) ---
    if isinstance(model, LogisticRegression):
        if SHAP_AVAILABLE and X_background is not None:
            try:
                explainer = shap.LinearExplainer(model, X_background)
                sv = np.array(explainer.shap_values(X_row), dtype=float).reshape(-1)
                out = pd.DataFrame({"Fitur": feature_columns, "Kontribusi": sv})
                out["abs"] = out["Kontribusi"].abs()
                return out.sort_values("abs", ascending=False).head(top_n), "shap"
            except Exception:
                pass
        # Fallback: kontribusi = koefisien * nilai fitur (sudah ter-scaling)
        coef = np.asarray(model.coef_).reshape(-1)
        x = np.asarray(X_row).reshape(-1)
        contrib = coef * x
        out = pd.DataFrame({"Fitur": feature_columns, "Kontribusi": contrib})
        out["abs"] = out["Kontribusi"].abs()
        return out.sort_values("abs", ascending=False).head(top_n), "linear_contrib"

    # --- Model tree-based ---
    if SHAP_AVAILABLE and isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)) or \
       (XGB_AVAILABLE and isinstance(model, xgb.XGBClassifier)):
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_row)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            sv = np.array(sv, dtype=float).reshape(-1)
            out = pd.DataFrame({"Fitur": feature_columns, "Kontribusi": sv})
            out["abs"] = out["Kontribusi"].abs()
            return out.sort_values("abs", ascending=False).head(top_n), "shap"
        except Exception:
            pass

    # --- Fallback umum: feature importance global ---
    imp = getattr(model, "feature_importances_", None)
    if imp is not None:
        out = pd.DataFrame({"Fitur": feature_columns, "Kontribusi": np.array(imp, dtype=float)})
        out["abs"] = out["Kontribusi"].abs()
        return out.sort_values("abs", ascending=False).head(top_n), "importance"
    return None, None


def plot_contrib(expl_df, mode):
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in expl_df["Kontribusi"]]
    fig = go.Figure(go.Bar(
        x=expl_df["Kontribusi"][::-1],
        y=expl_df["Fitur"][::-1],
        orientation="h",
        marker_color=colors[::-1],
    ))
    title_map = {
        "shap": "Kontribusi Fitur terhadap Prediksi (SHAP)",
        "linear_contrib": "Kontribusi Fitur terhadap Prediksi (Koefisien × Nilai)",
        "importance": "Feature Importance (global)",
    }
    fig.update_layout(
        title=title_map.get(mode, "Kontribusi Fitur"),
        height=420, margin=dict(t=50, b=20, l=10, r=10),
        xaxis_title="Dampak ke arah Resign (+) / Stay (-)",
    )
    return fig


# ============================================================
# 3.1 Inisialisasi Aplikasi & Sidebar Navigasi
# ============================================================
art = load_artifacts()
ref_df = load_reference_data()
schema = build_schema(ref_df) if ref_df is not None else None

st.sidebar.title("🛡️ AttritionGuard")
st.sidebar.caption("Employee Risk Prediction System")
page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda",
     "👤 Prediksi Individu",
     "📁 Prediksi Batch",
     "📊 Dashboard EDA",
     "ℹ️ Tentang"],
)

# Status artefak
with st.sidebar.expander("Status Sistem", expanded=False):
    st.write("Model:", "✅" if art["model"] is not None else "❌")
    st.write("Scaler:", "✅" if art["scaler"] is not None else "❌")
    st.write("Feature columns:", "✅" if art["feature_columns"] is not None else "❌")
    st.write("Dataset EDA:", "✅" if ref_df is not None else "❌")
    st.write("SHAP:", "✅" if SHAP_AVAILABLE else "❌ (pakai kontribusi koefisien)")
    st.write(f"Threshold: {art['threshold']:.2f}")
    for e in art["errors"]:
        st.warning(e)

MODEL_READY = art["model"] is not None and schema is not None

# Cache ringan untuk background data SHAP LinearExplainer (sample dari data referensi)
@st.cache_data(show_spinner=False)
def get_background_matrix(_ref_df, _schema, _feature_columns, _scaler, n=100):
    if _ref_df is None or _schema is None:
        return None
    sample = _ref_df.sample(min(n, len(_ref_df)), random_state=42)
    return preprocess(sample, _schema, _feature_columns, _scaler).values


# ============================================================
# 3.2 Halaman: Beranda
# ============================================================
if page.endswith("Beranda"):
    st.title("🛡️ AttritionGuard")
    st.subheader("Sistem Prediksi Risiko Attrition Karyawan")
    st.markdown(
        "AttritionGuard membantu perusahaan mengidentifikasi karyawan yang berpotensi "
        "resign sejak dini, sehingga tim HR dapat melakukan tindakan preventif untuk "
        "mempertahankan talenta terbaik."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Karyawan (data)", f"{len(ref_df):,}" if ref_df is not None else "-")
    if ref_df is not None and TARGET_COL in ref_df.columns:
        rate = (ref_df[TARGET_COL].astype(str).str.lower().isin(["yes", "1"])).mean()
        c2.metric("Tingkat Attrition", f"{rate*100:.1f}%")
    c3.metric("Threshold model", f"{art['threshold']:.2f}")
    c4.metric("Status model", "Siap" if MODEL_READY else "Belum")

    st.divider()
    st.markdown("#### Fitur Utama")
    f1, f2, f3 = st.columns(3)
    f1.info("**👤 Prediksi Individu**\n\nInput manual 1 karyawan + gauge risiko, penjelasan kontribusi fitur, dan rekomendasi.")
    f2.info("**📁 Prediksi Batch**\n\nUpload CSV banyak karyawan, ranking risiko, dan unduh hasil.")
    f3.info("**📊 Dashboard EDA**\n\nRingkasan insight & faktor pendorong attrition.")
    if not MODEL_READY:
        st.warning(
            "Model belum lengkap. Pastikan file berikut ada di folder **models/**: "
            "`best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `optimal_threshold.pkl`, "
            "dan dataset mentah di folder **data/**."
        )

# ============================================================
# 3.3 Halaman: Prediksi Individu
# ============================================================
elif page.endswith("Prediksi Individu"):
    st.title("👤 Prediksi Individu")
    st.markdown("Masukkan data satu karyawan untuk melihat probabilitas risiko resign beserta penjelasannya.")

    if not MODEL_READY:
        st.error("Model belum siap. Periksa status sistem di sidebar.")
    else:
        raw_cols = schema["raw_columns"]
        with st.form("form_individu"):
            st.markdown("#### Data Karyawan")
            cols_layout = st.columns(3)
            input_data = {}

            # Kolom numerik
            for i, col in enumerate(schema["num_cols"]):
                default = float(ref_df[col].median()) if ref_df is not None else 0.0
                lo = float(ref_df[col].min()) if ref_df is not None else 0.0
                hi = float(ref_df[col].max()) if ref_df is not None else default * 2 + 1
                with cols_layout[i % 3]:
                    input_data[col] = st.number_input(
                        col, value=default, min_value=lo, max_value=hi, step=1.0
                    )

            # Kolom binary
            for i, col in enumerate(schema["binary_cols"]):
                options = list(schema["binary_maps"][col].keys())
                with cols_layout[i % 3]:
                    input_data[col] = st.selectbox(col, options)

            # Kolom one-hot (kategorikal multi-nilai)
            for i, col in enumerate(schema["onehot_cols"]):
                options = sorted(ref_df[col].dropna().unique().tolist()) if ref_df is not None else []
                with cols_layout[i % 3]:
                    input_data[col] = st.selectbox(col, options)

            submitted = st.form_submit_button("🔍 Prediksi Risiko", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([input_data])
            X_processed = preprocess(input_df, schema, art["feature_columns"], art["scaler"])
            prob = float(predict_proba(art["model"], X_processed)[0])
            band, color = risk_band(prob)

            st.divider()
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.plotly_chart(gauge_chart(prob), use_container_width=True)
                pred_label = "Berisiko Resign" if prob >= art["threshold"] else "Cenderung Stay"
                st.markdown(
                    f"**Prediksi (threshold {art['threshold']:.2f}):** "
                    f":{'red' if prob >= art['threshold'] else 'green'}[{pred_label}]"
                )

            with res_col2:
                st.markdown("#### Kontribusi Fitur terhadap Prediksi")
                background = get_background_matrix(ref_df, schema, art["feature_columns"], art["scaler"])
                expl_df, mode = shap_explanation(
                    art["model"], X_processed.values, art["feature_columns"],
                    X_background=background, top_n=10
                )
                if expl_df is not None:
                    st.plotly_chart(plot_contrib(expl_df, mode), use_container_width=True)
                else:
                    st.info("Penjelasan kontribusi fitur tidak tersedia untuk model ini.")

            st.divider()
            st.markdown("#### Rekomendasi Tindakan untuk HR")
            recs = build_recommendations(input_data, ref_df)
            for judul, saran in recs:
                st.warning(f"**{judul}** — {saran}")

# ============================================================
# 3.4 Halaman: Prediksi Batch
# ============================================================
elif page.endswith("Prediksi Batch"):
    st.title("📁 Prediksi Batch")
    st.markdown(
        "Upload file CSV berisi data banyak karyawan (format kolom sama seperti dataset asli) "
        "untuk memprediksi risiko resign sekaligus dan mengurutkan berdasarkan tingkat risiko."
    )

    if not MODEL_READY:
        st.error("Model belum siap. Periksa status sistem di sidebar.")
    else:
        if ref_df is not None:
            template_csv = ref_df[schema["raw_columns"]].head(5).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Unduh Template CSV (contoh format)",
                data=template_csv,
                file_name="template_prediksi_batch.csv",
                mime="text/csv",
            )

        uploaded = st.file_uploader("Upload file CSV", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")
                batch_df = None

            if batch_df is not None:
                missing_cols = [c for c in schema["raw_columns"] if c not in batch_df.columns]
                if missing_cols:
                    st.warning(f"Kolom berikut tidak ditemukan dan akan diisi nilai default: {missing_cols}")

                with st.spinner("Memproses prediksi..."):
                    X_batch = preprocess(batch_df, schema, art["feature_columns"], art["scaler"])
                    probs = predict_proba(art["model"], X_batch.values)

                result_df = batch_df.copy()
                result_df["Probabilitas_Resign"] = (probs * 100).round(2)
                result_df["Kategori_Risiko"] = [risk_band(p)[0] for p in probs]
                result_df["Prediksi"] = np.where(
                    probs >= art["threshold"], "Berisiko Resign", "Cenderung Stay"
                )
                result_df = result_df.sort_values("Probabilitas_Resign", ascending=False)

                st.success(f"✅ Prediksi selesai untuk {len(result_df)} karyawan.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Karyawan", f"{len(result_df):,}")
                m2.metric("Risiko Tinggi", int((result_df["Kategori_Risiko"] == "Tinggi").sum()))
                m3.metric("Risiko Sedang", int((result_df["Kategori_Risiko"] == "Sedang").sum()))

                st.markdown("#### Ranking Risiko Karyawan")

                def highlight_risk(val):
                    color_map = {"Tinggi": "#fdedec", "Sedang": "#fef5e7", "Rendah": "#eafaf1"}
                    return f"background-color: {color_map.get(val, '')}"

                st.dataframe(
                    result_df.style.applymap(highlight_risk, subset=["Kategori_Risiko"]),
                    use_container_width=True,
                    height=420,
                )

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Unduh Hasil Prediksi (CSV)",
                    data=csv_out,
                    file_name="hasil_prediksi_attrition.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.divider()
                fig_dist = px.histogram(
                    result_df, x="Probabilitas_Resign", color="Kategori_Risiko",
                    color_discrete_map={"Tinggi": "#e74c3c", "Sedang": "#f39c12", "Rendah": "#2ecc71"},
                    nbins=20, title="Distribusi Probabilitas Resign — Seluruh Karyawan",
                )
                st.plotly_chart(fig_dist, use_container_width=True)

# ============================================================
# 3.5 Halaman: Dashboard EDA
# ============================================================
elif page.endswith("Dashboard EDA"):
    st.title("Dashboard EDA")
    st.markdown("Ringkasan temuan eksplorasi data dari tahap analisis (lihat notebook EDA untuk detail lengkap).")

    if ref_df is None:
        st.error("Dataset referensi tidak ditemukan.")
    else:
        df = ref_df.copy()
        df["Attrition_Flag"] = df[TARGET_COL].astype(str).str.lower().isin(["yes", "1"]).astype(int)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Karyawan", f"{len(df):,}")
        m2.metric("Tingkat Attrition", f"{df['Attrition_Flag'].mean()*100:.1f}%")
        m3.metric("Rata-rata Usia", f"{df['Age'].mean():.1f} thn" if "Age" in df else "-")
        m4.metric("Rata-rata Gaji", f"Rp{df['MonthlyIncome'].mean():,.0f}" if "MonthlyIncome" in df else "-")

        st.divider()
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Demografi", "Faktor Pekerjaan", "Kompensasi & Kepuasan", "Korelasi"]
        )

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                if "Age" in df.columns:
                    fig = px.histogram(
                        df, x="Age", color=df["Attrition_Flag"].map({0: "Stay", 1: "Resign"}),
                        nbins=20, barmode="overlay",
                        color_discrete_map={"Stay": "#3498db", "Resign": "#e74c3c"},
                        title="Distribusi Usia berdasarkan Status Attrition",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if "Gender" in df.columns:
                    rate = df.groupby("Gender")["Attrition_Flag"].mean().reset_index()
                    rate["Attrition_Flag"] *= 100
                    fig = px.bar(
                        rate, x="Gender", y="Attrition_Flag",
                        title="Tingkat Attrition berdasarkan Gender (%)",
                        color_discrete_sequence=["#3498db"],
                    )
                    fig.update_layout(yaxis_title="Tingkat Attrition (%)")
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                if "Department" in df.columns:
                    rate = df.groupby("Department")["Attrition_Flag"].mean().reset_index()
                    rate["Attrition_Flag"] *= 100
                    fig = px.bar(
                        rate.sort_values("Attrition_Flag", ascending=False),
                        x="Department", y="Attrition_Flag",
                        title="Tingkat Attrition per Departemen (%)",
                        color_discrete_sequence=["#e74c3c"],
                    )
                    fig.update_layout(yaxis_title="Tingkat Attrition (%)")
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if "OverTime" in df.columns:
                    rate = df.groupby("OverTime")["Attrition_Flag"].mean().reset_index()
                    rate["Attrition_Flag"] *= 100
                    fig = px.bar(
                        rate, x="OverTime", y="Attrition_Flag",
                        title="Tingkat Attrition: OverTime vs Tidak (%)",
                        color_discrete_sequence=["#f39c12"],
                    )
                    fig.update_layout(yaxis_title="Tingkat Attrition (%)")
                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                if "MonthlyIncome" in df.columns:
                    fig = px.box(
                        df, x=df["Attrition_Flag"].map({0: "Stay", 1: "Resign"}), y="MonthlyIncome",
                        title="Monthly Income berdasarkan Status Attrition",
                        color=df["Attrition_Flag"].map({0: "Stay", 1: "Resign"}),
                        color_discrete_map={"Stay": "#3498db", "Resign": "#e74c3c"},
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                sat_cols = [c for c in ["JobSatisfaction", "EnvironmentSatisfaction",
                                        "RelationshipSatisfaction", "WorkLifeBalance"] if c in df.columns]
                if sat_cols:
                    means = df.groupby("Attrition_Flag")[sat_cols].mean().T.reset_index()
                    means.columns = ["Indikator", "Stay", "Resign"]
                    fig = px.bar(
                        means.melt(id_vars="Indikator", var_name="Status", value_name="Skor"),
                        x="Indikator", y="Skor", color="Status", barmode="group",
                        title="Rata-rata Indikator Kepuasan: Stay vs Resign",
                        color_discrete_map={"Stay": "#3498db", "Resign": "#e74c3c"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with tab4:
            num_df = df.select_dtypes(include="number")
            if "Attrition_Flag" in num_df.columns:
                corr = num_df.corr()["Attrition_Flag"].drop("Attrition_Flag").sort_values(
                    key=abs, ascending=False
                ).head(15)
                fig = px.bar(
                    x=corr.values, y=corr.index, orientation="h",
                    title="Top 15 Korelasi Fitur dengan Attrition",
                    color=corr.values,
                    color_continuous_scale=["#3498db", "#ecf0f1", "#e74c3c"],
                )
                fig.update_layout(xaxis_title="Korelasi", yaxis_title="", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Ringkasan Insight Utama")
        insight_table = pd.DataFrame([
            ["OverTime / Is_Overworked", "Karyawan yang lembur resign ~3x lebih banyak (30.5% vs 10.4%)"],
            ["Job Level", "Entry-level (Level 1) paling rentan: attrition 26.3%"],
            ["Job Role", "Sales Representative tertinggi (39.8%), Research Director terendah (2.5%)"],
            ["Monthly Income", "Karyawan resign berpenghasilan rata-rata 29% lebih rendah"],
            ["Status Pernikahan", "Single: 25.5% vs menikah: 11.7%"],
            ["Generasi", "Millennial paling rentan (20.2%)"],
            ["Overall Satisfaction", "Stay: 2.76 vs Resign: 2.51"],
            ["Departemen", "Sales paling rentan (20.6%), R&D paling stabil (13.8%)"],
            ["Pengalaman Kerja", "Karyawan baru & tahun jabatan rendah lebih berisiko"],
            ["Business Travel", "Sering perjalanan bisnis meningkatkan risiko resign"],
        ], columns=["Faktor", "Temuan"])
        st.table(insight_table)

# ============================================================
# 3.6 Halaman: Tentang
# ============================================================
elif page.endswith("Tentang"):
    st.title("Tentang AttritionGuard")

    st.markdown("""
    **AttritionGuard** adalah sistem prediksi risiko attrition (resign) karyawan
    yang dibangun di atas dataset **IBM HR Analytics Employee Attrition & Performance**
    (1.470 baris data karyawan).

    Aplikasi ini dikembangkan secara bertahap oleh tim, terdiri dari:
    - **Data Preparation** — Zahra Daniah (cleaning, preprocessing, feature engineering)
    - **Exploratory Data Analysis** — Nailah Fauziyyah (analisis & insight)
    - **Machine Learning** — Hafizatul Khairani (pemilihan, training, evaluasi, tuning model)
    - **Deployment** — Nabila Nur Aini (aplikasi web Streamlit ini)
    """)

    st.divider()
    st.markdown("#### Informasi Model")

    if art["model"] is not None:
        model_name = type(art["model"]).__name__
        c1, c2, c3 = st.columns(3)
        c1.metric("Algoritma", model_name)
        c2.metric("Jumlah Fitur", len(art["feature_columns"]) if art["feature_columns"] else "-")
        c3.metric("Threshold Optimal", f"{art['threshold']:.2f}")

        st.markdown(f"""
        **Mengapa threshold {art['threshold']:.2f}, bukan 0.5?**

        Threshold default (0.5) menghasilkan recall yang rendah — banyak karyawan yang
        sebenarnya akan resign tidak tertangkap oleh model. Threshold diturunkan ke
        **{art['threshold']:.2f}** untuk memaksimalkan **recall ≥ 70%**, dengan
        konsekuensi jumlah *false alarm* (karyawan ditandai berisiko padahal sebenarnya
        bertahan) menjadi lebih banyak.

        Trade-off ini diambil secara sengaja: dalam konteks retensi karyawan,
        **biaya kehilangan talenta (false negative) jauh lebih mahal** dibanding
        biaya melakukan intervensi pencegahan ke karyawan yang sebenarnya tidak akan resign
        (false positive).
        """)
    else:
        st.warning("Model belum dimuat.")

    st.divider()
    st.markdown("#### Performa Model (Hasil Evaluasi pada Test Set)")
    perf_table = pd.DataFrame([
        ["Logistic Regression", 0.8435, 0.5135, 0.4043, 0.4524, 0.8022],
        ["XGBoost", 0.8673, 0.7000, 0.2979, 0.4179, 0.7655],
        ["Decision Tree", 0.7279, 0.2676, 0.4043, 0.3220, 0.5969],
        ["Random Forest", 0.8299, 0.4348, 0.2128, 0.2857, 0.7391],
    ], columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"])
    st.dataframe(perf_table, use_container_width=True, hide_index=True)
    st.caption(
        "Logistic Regression dipilih sebagai model produksi karena F1-Score dan AUC-ROC "
        "tertinggi serta paling stabil pada cross-validation (mean F1 = 0.894, std = 0.020)."
    )

    st.divider()
    st.markdown("#### Batasan & Disclaimer")
    st.warning("""
    - Model dilatih pada data historis IBM HR Analytics dan **belum tentu merepresentasikan**
      kondisi perusahaan lain secara langsung.
    - Recall model (~72% pada threshold optimal) berarti **sekitar 1 dari 4 karyawan**
      yang akan resign tidak terdeteksi oleh sistem.
    - Precision yang moderat berarti **akan ada false alarm** — tidak semua karyawan yang
      ditandai berisiko benar-benar akan resign.
    - **Prediksi ini adalah alat bantu pendukung keputusan**, bukan keputusan final.
      Keputusan HR (mutasi, kenaikan gaji, dsb.) tetap harus mempertimbangkan konteks
      manusia dan bukti tambahan di luar model.
    """)

    st.divider()
    st.caption("AttritionGuard v1.0 — Dataset: IBM HR Analytics Employee Attrition & Performance (Kaggle)")
