"""
AttritionGuard - Employee Risk Prediction System
Bagian Deployment (Streamlit)

Fitur:
  1. Prediksi Individu (input manual 1 karyawan)
  2. Prediksi Batch (upload CSV banyak karyawan)
  3. Dashboard Ringkasan EDA / Insight
  4. Penjelasan SHAP / Feature Importance per prediksi
  5. Prediksi Risiko Attrition + Rekomendasi Tindakan

Catatan arsitektur:
  - Praproses (encoding + feature engineering + scaling) DIREPLIKASI persis
    dari notebook Data Preparation (Zahra) agar konsisten dengan model.
  - Penyelarasan kolom memakai models/feature_columns.pkl dan kolom yang
    di-scale diambil dari scaler.feature_names_in_ -> robust terhadap
    perbedaan kategori one-hot.
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

# ----------------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AttritionGuard - Prediksi Risiko Attrition",
    page_icon="\U0001F6E1\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Lokasi file (relatif terhadap struktur repo)
# ----------------------------------------------------------------------------
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
        # fallback: pola nama
    return None


# ----------------------------------------------------------------------------
# Loader artefak model
# ----------------------------------------------------------------------------
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
    # kumpulkan kandidat: nama preferensi dulu, lalu csv lain di folder data
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
    # pilih yang benar-benar mentah lebih dulu
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


# ----------------------------------------------------------------------------
# Praproses: replikasi pipeline notebook
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Rekomendasi tindakan (rule-based dari insight EDA)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Fitur penting untuk form ringkas (Prediksi Individu)
# Dipilih berdasarkan Ringkasan Insight EDA (10 faktor utama attrition).
# Kolom lain tetap dikirim ke model, tapi diisi otomatis dgn median/mode.
# ----------------------------------------------------------------------------
KEY_INPUT_COLS = [
    "OverTime", "JobLevel", "JobRole", "MonthlyIncome", "MaritalStatus",
    "Department", "JobSatisfaction", "BusinessTravel", "TotalWorkingYears", "Age",
]


def default_inputs(schema, ref_df):
    """Nilai default (median utk numerik, mode utk kategorikal) utk semua raw_columns."""
    defaults = {}
    for col in schema["raw_columns"]:
        series = ref_df[col].dropna()
        if col in schema["cat_cols"]:
            defaults[col] = str(series.mode().iloc[0]) if not series.empty else ""
        else:
            is_int = pd.api.types.is_integer_dtype(ref_df[col])
            val = series.median() if not series.empty else 0
            defaults[col] = int(val) if is_int else float(val)
    return defaults


def sample_row(ref_df, schema, risk="random"):
    """Ambil satu baris contoh dari dataset referensi untuk mengisi form otomatis.
    risk: 'high' (cenderung berisiko), 'low' (cenderung aman), atau 'random'."""
    df = ref_df.copy()
    if risk == "high" and "OverTime" in df.columns:
        mask = df["OverTime"].astype(str).str.lower().isin(["yes", "1", "true"])
        pool = df[mask] if mask.any() else df
    elif risk == "low" and "OverTime" in df.columns:
        mask = ~df["OverTime"].astype(str).str.lower().isin(["yes", "1", "true"])
        pool = df[mask] if mask.any() else df
    else:
        pool = df
    row = pool.sample(1).iloc[0]
    result = {}
    for col in schema["raw_columns"]:
        if col in schema["cat_cols"]:
            result[col] = str(row[col])
        else:
            is_int = pd.api.types.is_integer_dtype(ref_df[col])
            result[col] = int(row[col]) if is_int else float(row[col])
    return result


# ----------------------------------------------------------------------------
# Komponen UI
# ----------------------------------------------------------------------------
def gauge_chart(prob):
    band, color = risk_band(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": f"Probabilitas Resign &mdash; Risiko {band}"},
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


def shap_explanation(model, X_row, feature_columns, top_n=10):
    """Hitung kontribusi fitur untuk 1 prediksi. Return DataFrame (fitur, shap)."""
    if SHAP_AVAILABLE:
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
    # Fallback: feature importance global
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
    title = ("Kontribusi Fitur terhadap Prediksi (SHAP)" if mode == "shap"
             else "Feature Importance (global)")
    fig.update_layout(title=title, height=420, margin=dict(t=50, b=20, l=10, r=10),
                      xaxis_title="Dampak ke arah Resign (+) / Stay (-)")
    return fig


# ============================================================================
# APLIKASI
# ============================================================================
art = load_artifacts()
ref_df = load_reference_data()
schema = build_schema(ref_df) if ref_df is not None else None

st.sidebar.title("\U0001F6E1\uFE0F AttritionGuard")
st.sidebar.caption("Employee Risk Prediction System")
page = st.sidebar.radio(
    "Navigasi",
    ["\U0001F3E0 Beranda",
     "\U0001F464 Prediksi Individu",
     "\U0001F4C1 Prediksi Batch",
     "\U0001F4CA Dashboard EDA",
     "\u2139\uFE0F Tentang"],
)

# Status artefak
with st.sidebar.expander("Status Sistem", expanded=False):
    st.write("Model:", "\u2705" if art["model"] is not None else "\u274C")
    st.write("Scaler:", "\u2705" if art["scaler"] is not None else "\u274C")
    st.write("Feature columns:", "\u2705" if art["feature_columns"] is not None else "\u274C")
    st.write("Dataset EDA:", "\u2705" if ref_df is not None else "\u274C")
    st.write("SHAP:", "\u2705" if SHAP_AVAILABLE else "\u274C (pakai importance)")
    st.write(f"Threshold: {art['threshold']:.2f}")
    for e in art["errors"]:
        st.warning(e)

MODEL_READY = art["model"] is not None and schema is not None


# ----------------------------------------------------------------------------
# HALAMAN: BERANDA
# ----------------------------------------------------------------------------
if page.endswith("Beranda"):
    st.title("\U0001F6E1\uFE0F AttritionGuard")
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
    f1.info("**\U0001F464 Prediksi Individu**\n\nInput manual 1 karyawan + gauge risiko, penjelasan SHAP, dan rekomendasi.")
    f2.info("**\U0001F4C1 Prediksi Batch**\n\nUpload CSV banyak karyawan, ranking risiko, dan unduh hasil.")
    f3.info("**\U0001F4CA Dashboard EDA**\n\nRingkasan insight & faktor pendorong attrition.")
    if not MODEL_READY:
        st.warning(
            "Model belum lengkap. Pastikan file berikut ada di folder **models/**: "
            "`best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `optimal_threshold.pkl`, "
            "dan dataset mentah di folder **data/**."
        )


# ----------------------------------------------------------------------------
# HALAMAN: PREDIKSI INDIVIDU
# ----------------------------------------------------------------------------
elif page.endswith("Prediksi Individu"):
    st.title("\U0001F464 Prediksi Individu")
    st.caption("Masukkan data 1 karyawan untuk memprediksi risiko attrition. "
               "Hanya 10 faktor paling berpengaruh (hasil EDA) yang perlu diisi manual; "
               "faktor lain otomatis terisi nilai tipikal dari dataset.")

    if not MODEL_READY:
        st.error("Model atau dataset referensi belum tersedia. Lihat panel Status Sistem.")
        st.stop()

    raw_cols = schema["raw_columns"]

    # --- Tombol contoh cepat (untuk demo) ---
    st.markdown("**Isi cepat dengan contoh dari dataset:**")
    c1, c2, c3 = st.columns(3)
    if c1.button("\U0001F534 Contoh Risiko Tinggi", use_container_width=True):
        st.session_state["form_values"] = sample_row(ref_df, schema, risk="high")
    if c2.button("\U0001F7E2 Contoh Risiko Rendah", use_container_width=True):
        st.session_state["form_values"] = sample_row(ref_df, schema, risk="low")
    if c3.button("\U0001F3B2 Acak dari Dataset", use_container_width=True):
        st.session_state["form_values"] = sample_row(ref_df, schema, risk="random")

    base_values = st.session_state.get("form_values") or default_inputs(schema, ref_df)

    inputs = dict(base_values)  # mulai dari default/contoh; field kunci akan ditimpa input user
    with st.form("form_individu"):
        st.caption("\U0001F511 10 faktor utama (Insight EDA)")
        key_cols = [c for c in KEY_INPUT_COLS if c in raw_cols]
        cols = st.columns(3)
        for i, col in enumerate(key_cols):
            target = cols[i % 3]
            if col in schema["cat_cols"]:
                opts = sorted([str(x) for x in ref_df[col].dropna().unique()])
                default_idx = opts.index(str(base_values[col])) if str(base_values[col]) in opts else 0
                inputs[col] = target.selectbox(col, opts, index=default_idx)
            else:
                series = ref_df[col].dropna()
                vmin, vmax = float(series.min()), float(series.max())
                is_int = pd.api.types.is_integer_dtype(ref_df[col])
                step = 1.0 if is_int else max((vmax - vmin) / 100, 0.01)
                val = target.number_input(col, min_value=vmin, max_value=vmax,
                                          value=float(base_values[col]), step=step)
                inputs[col] = int(val) if is_int else val

        with st.expander("\u2699\ufe0f Faktor lain (opsional — terisi otomatis, bisa disesuaikan)"):
            other_cols = [c for c in raw_cols if c not in KEY_INPUT_COLS]
            cols2 = st.columns(3)
            for i, col in enumerate(other_cols):
                target = cols2[i % 3]
                if col in schema["cat_cols"]:
                    opts = sorted([str(x) for x in ref_df[col].dropna().unique()])
                    default_idx = opts.index(str(base_values[col])) if str(base_values[col]) in opts else 0
                    inputs[col] = target.selectbox(col, opts, index=default_idx, key=f"other_{col}")
                else:
                    series = ref_df[col].dropna()
                    vmin, vmax = float(series.min()), float(series.max())
                    is_int = pd.api.types.is_integer_dtype(ref_df[col])
                    step = 1.0 if is_int else max((vmax - vmin) / 100, 0.01)
                    val = target.number_input(col, min_value=vmin, max_value=vmax,
                                              value=float(base_values[col]), step=step, key=f"other_{col}")
                    inputs[col] = int(val) if is_int else val

        submitted = st.form_submit_button("\U0001F50D Prediksi", use_container_width=True)

    if submitted:
        raw_row = pd.DataFrame([inputs])
        X = preprocess(raw_row, schema, art["feature_columns"], art["scaler"])
        prob = float(predict_proba(art["model"], X)[0])
        band, color = risk_band(prob)
        pred_label = "BERISIKO RESIGN" if prob >= art["threshold"] else "CENDERUNG BERTAHAN"

        st.divider()
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)
        with right:
            st.markdown(f"### Status: <span style='color:{color}'>{pred_label}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**Probabilitas resign:** {prob*100:.1f}%")
            st.markdown(f"**Tingkat risiko:** :{ 'red' if band=='Tinggi' else ('orange' if band=='Sedang' else 'green')}[{band}]")
            st.progress(min(prob, 1.0))

        st.divider()
        tab1, tab2 = st.tabs(["\U0001F9E0 Penjelasan (SHAP)", "\U0001F4A1 Rekomendasi Tindakan"])
        with tab1:
            expl, mode = shap_explanation(art["model"], X.values,
                                          art["feature_columns"] or list(X.columns))
            if expl is not None:
                st.plotly_chart(plot_contrib(expl, mode), use_container_width=True)
                st.caption("Merah = mendorong ke arah resign, Biru = menahan (cenderung bertahan)."
                           if mode == "shap" else
                           "Menampilkan feature importance global (SHAP tidak tersedia).")
            else:
                st.info("Penjelasan fitur tidak tersedia untuk model ini.")
        with tab2:
            for faktor, aksi in build_recommendations(inputs, ref_df):
                st.markdown(f"**{faktor}**  \n{aksi}")


# ----------------------------------------------------------------------------
# HALAMAN: PREDIKSI BATCH
# ----------------------------------------------------------------------------
elif page.endswith("Prediksi Batch"):
    st.title("\U0001F4C1 Prediksi Batch")
    st.caption("Upload file CSV berisi banyak karyawan untuk prediksi sekaligus.")

    if not MODEL_READY:
        st.error("Model atau dataset referensi belum tersedia. Lihat panel Status Sistem.")
        st.stop()

    # Template CSV
    tmpl = ref_df.drop(columns=[c for c in [TARGET_COL] if c in ref_df.columns]).head(5)
    st.download_button(
        "\U0001F4C4 Unduh template CSV",
        tmpl.to_csv(index=False).encode("utf-8"),
        file_name="template_input_karyawan.csv",
        mime="text/csv",
    )

    up = st.file_uploader("Upload CSV karyawan", type=["csv"])
    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            st.stop()
        st.write(f"Jumlah baris: **{len(data)}**")
        st.dataframe(data.head(), use_container_width=True)

        if st.button("\U0001F680 Jalankan Prediksi Batch", use_container_width=True):
            X = preprocess(data, schema, art["feature_columns"], art["scaler"])
            probs = predict_proba(art["model"], X)
            res = data.copy()
            res["Prob_Resign"] = (probs * 100).round(1)
            res["Prediksi"] = np.where(probs >= art["threshold"], "Berisiko Resign", "Bertahan")
            res["Tingkat_Risiko"] = [risk_band(p)[0] for p in probs]
            res = res.sort_values("Prob_Resign", ascending=False)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total karyawan", len(res))
            c2.metric("Berisiko resign", int((probs >= art["threshold"]).sum()))
            c3.metric("Risiko tinggi", int(sum(risk_band(p)[0] == "Tinggi" for p in probs)))

            dist = pd.Series([risk_band(p)[0] for p in probs]).value_counts()
            fig = px.pie(values=dist.values, names=dist.index,
                         color=dist.index,
                         color_discrete_map={"Rendah": "#2ecc71", "Sedang": "#f39c12",
                                             "Tinggi": "#e74c3c"},
                         title="Distribusi Tingkat Risiko")
            st.plotly_chart(fig, use_container_width=True)

            def _hl(v):
                c = {"Tinggi": "#fdedec", "Sedang": "#fef5e7", "Rendah": "#eafaf1"}.get(v, "")
                return f"background-color: {c}"
            st.dataframe(res.style.applymap(_hl, subset=["Tingkat_Risiko"]),
                         use_container_width=True, height=420)

            st.download_button(
                "\U0001F4BE Unduh hasil prediksi (CSV)",
                res.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_attrition.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ----------------------------------------------------------------------------
# HALAMAN: DASHBOARD EDA
# ----------------------------------------------------------------------------
elif page.endswith("Dashboard EDA"):
    st.title("\U0001F4CA Dashboard Ringkasan EDA & Insight")
    if ref_df is None:
        st.error("Dataset tidak ditemukan di folder data/.")
        st.stop()

    df = ref_df.copy()
    has_target = TARGET_COL in df.columns
    if has_target:
        df["_resign"] = df[TARGET_COL].astype(str).str.lower().isin(["yes", "1"]).astype(int)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total karyawan", f"{len(df):,}")
    if has_target:
        c2.metric("Tingkat attrition", f"{df['_resign'].mean()*100:.1f}%")
        c3.metric("Jumlah resign", int(df["_resign"].sum()))

    st.divider()

    def rate_by(col):
        g = df.groupby(col)["_resign"].mean().sort_values(ascending=False) * 100
        return g

    if has_target:
        colA, colB = st.columns(2)
        for col, container in [("OverTime", colA), ("MaritalStatus", colB),
                               ("Department", colA), ("JobRole", colB),
                               ("BusinessTravel", colA), ("JobLevel", colB)]:
            if col in df.columns:
                g = rate_by(col)
                fig = px.bar(x=g.index.astype(str), y=g.values,
                             labels={"x": col, "y": "Attrition Rate (%)"},
                             title=f"Attrition berdasarkan {col}",
                             color=g.values, color_continuous_scale="Reds")
                container.plotly_chart(fig, use_container_width=True)

        if "MonthlyIncome" in df.columns:
            fig = px.box(df, x=TARGET_COL, y="MonthlyIncome", color=TARGET_COL,
                         title="Distribusi Monthly Income: Resign vs Bertahan")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Kolom target tidak ada pada dataset, menampilkan distribusi umum.")
        num = df.select_dtypes(include=["int64", "float64"]).columns[:6]
        for col in num:
            st.plotly_chart(px.histogram(df, x=col, title=f"Distribusi {col}"),
                            use_container_width=True)

    st.divider()
    st.markdown("### \U0001F511 Faktor Utama Pendorong Attrition")
    st.markdown(
        """
| # | Faktor | Temuan |
|---|--------|--------|
| 1 | **OverTime** | Karyawan lembur resign ~3x lebih banyak (30.5% vs 10.4%) |
| 2 | **Job Level** | Entry-level (Level 1) paling rentan: 26.3% |
| 3 | **Job Role** | Sales Representative tertinggi (39.8%) |
| 4 | **Monthly Income** | Karyawan resign berpenghasilan ~29% lebih rendah |
| 5 | **Status Pernikahan** | Single 25.5% vs menikah 11.7% |
| 6 | **Generasi** | Millennial paling rentan (20.2%) |
| 7 | **Business Travel** | Sering dinas meningkatkan risiko resign |
"""
    )
    st.markdown("### \U0001F4A1 Rekomendasi Strategis")
    st.markdown(
        "1. **Batasi OverTime** dan pantau beban kerja.\n"
        "2. **Program retensi entry-level** (mentoring, jalur karier, kompensasi).\n"
        "3. **Perhatikan role bertekanan tinggi** seperti Sales Representative.\n"
        "4. **Review kompensasi** untuk menutup gap gaji.\n"
        "5. **Engagement untuk karyawan muda/single**."
    )


# ----------------------------------------------------------------------------
# HALAMAN: TENTANG
# ----------------------------------------------------------------------------
else:
    st.title("\u2139\uFE0F Tentang AttritionGuard")
    st.markdown(
        """
**AttritionGuard** adalah sistem prediksi risiko attrition karyawan berbasis Machine Learning,
dibangun untuk **GWE 2026 Data Science Challenge**.

**Pipeline:**
1. Data Preparation \u2192 cleaning, encoding, feature engineering, scaling
2. EDA \u2192 analisis faktor pendorong attrition
3. Machine Learning \u2192 Logistic Regression, Decision Tree, Random Forest, XGBoost + SMOTE
4. **Deployment (halaman ini)** \u2192 Streamlit

**Fitur deployment:** prediksi individu, prediksi batch, dashboard EDA,
penjelasan SHAP per prediksi, serta rekomendasi tindakan.

**Tech stack:** Python, scikit-learn, XGBoost, SHAP, Pandas, Plotly, Streamlit.
"""
    )
    st.caption("GWE 2026 Data Science Challenge | Grow With EDM Gen 7")
