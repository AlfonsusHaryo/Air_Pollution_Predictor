import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# GLOBAL CONFIG
# =========================================================
APP_TITLE = "Air Quality Intelligence"
APP_SUBTITLE = "Prediksi Kategori Kualitas Udara Berbasis Machine Learning"

FEATURES = [
    "CO AQI Value",
    "Ozone AQI Value",
    "NO2 AQI Value",
    "PM2.5 AQI Value",
]

AQI_ORDER = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

AQI_COLORS = {
    "Good": "#22c55e",
    "Moderate": "#eab308",
    "Unhealthy for Sensitive Groups": "#f97316",
    "Unhealthy": "#ef4444",
    "Very Unhealthy": "#a855f7",
    "Hazardous": "#7f1d1d",
}

AQI_ADVICE = {
    "Good": {
        "status": "Udara baik dan aman untuk aktivitas luar ruangan.",
        "advice": "Aktivitas normal dapat dilakukan. Tetap jaga pola hidup sehat.",
        "risk": "Rendah",
    },
    "Moderate": {
        "status": "Kualitas udara masih dapat diterima, tetapi mulai perlu diperhatikan.",
        "advice": "Kelompok sensitif sebaiknya mengurangi aktivitas luar ruangan yang terlalu lama.",
        "risk": "Ringan",
    },
    "Unhealthy for Sensitive Groups": {
        "status": "Mulai tidak sehat untuk kelompok sensitif.",
        "advice": "Anak-anak, lansia, dan orang dengan gangguan pernapasan sebaiknya membatasi aktivitas outdoor.",
        "risk": "Sedang",
    },
    "Unhealthy": {
        "status": "Tidak sehat untuk masyarakat umum.",
        "advice": "Kurangi aktivitas luar ruangan. Gunakan masker jika harus keluar.",
        "risk": "Tinggi",
    },
    "Very Unhealthy": {
        "status": "Sangat tidak sehat dan dapat berdampak serius.",
        "advice": "Hindari aktivitas luar ruangan. Gunakan purifier atau tetap berada di dalam ruangan jika memungkinkan.",
        "risk": "Sangat tinggi",
    },
    "Hazardous": {
        "status": "Berbahaya dan membutuhkan kewaspadaan tinggi.",
        "advice": "Tetap di dalam ruangan, tutup ventilasi, gunakan masker berkualitas jika harus keluar.",
        "risk": "Kritis",
    },
}

MODEL_DIR = Path("model")
FALLBACK_OUTPUT_DIR = Path("outputs_no_tuning")


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 45%, #f8fafc 100%);
    }

    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        padding: 2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0369a1 100%);
        color: white;
        box-shadow: 0 22px 60px rgba(15, 23, 42, 0.22);
        margin-bottom: 1.4rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.04em;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        max-width: 900px;
        line-height: 1.65;
    }

    .soft-card {
        padding: 1.3rem 1.35rem;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.23);
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.07);
        margin-bottom: 1rem;
    }

    .prediction-card {
        padding: 1.6rem;
        border-radius: 28px;
        color: white;
        box-shadow: 0 20px 48px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }

    .small-label {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }

    .big-number {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.04em;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 20px;
        background: #ffffff;
        border: 1px solid rgba(226, 232, 240, 1);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        height: 100%;
    }

    .feature-pill {
        display: inline-block;
        padding: 0.42rem 0.7rem;
        margin: 0.18rem;
        background: #e0f2fe;
        color: #075985;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 700;
    }

    .footer-note {
        color: #64748b;
        font-size: 0.85rem;
        line-height: 1.55;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 10px 18px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
    }

    .stTabs [aria-selected="true"] {
        background: #0f172a !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def find_asset(filename: str) -> Optional[Path]:
    """Mencari file model pada folder model/ atau outputs_no_tuning/."""
    candidates = [
        MODEL_DIR / filename,
        FALLBACK_OUTPUT_DIR / filename,
        Path(filename),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[object, object, Dict]:
    """Load model, scaler, dan metadata hasil export notebook."""
    model_path = find_asset("best_model_no_tuning.pkl")
    scaler_path = find_asset("scaler_no_tuning.pkl")
    meta_path = find_asset("model_metadata_no_tuning.json")

    missing = []
    if model_path is None:
        missing.append("best_model_no_tuning.pkl")
    if scaler_path is None:
        missing.append("scaler_no_tuning.pkl")
    if meta_path is None:
        missing.append("model_metadata_no_tuning.json")

    if missing:
        raise FileNotFoundError(
            "File model belum ditemukan: "
            + ", ".join(missing)
            + ". Letakkan file tersebut di folder model/ atau jalankan export_model_from_csv.py."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(meta_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, scaler, metadata


def normalize_class_mapping(metadata: Dict) -> Dict[int, str]:
    """Metadata JSON menyimpan key dictionary sebagai string, jadi perlu dinormalisasi."""
    raw_mapping = metadata.get("class_mapping", {})
    mapping = {}
    for key, value in raw_mapping.items():
        try:
            mapping[int(key)] = value
        except Exception:
            mapping[key] = value

    if not mapping:
        mapping = {idx: label for idx, label in enumerate(AQI_ORDER)}

    return mapping


def build_input_dataframe(co: float, ozone: float, no2: float, pm25: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "CO AQI Value": co,
                "Ozone AQI Value": ozone,
                "NO2 AQI Value": no2,
                "PM2.5 AQI Value": pm25,
            }
        ]
    )


def predict_category(model, scaler, metadata: Dict, input_df: pd.DataFrame) -> Dict:
    """Melakukan prediksi satu baris input."""
    features = metadata.get("features", FEATURES)
    if isinstance(features, str):
        features = [item.strip() for item in features.split(",")]

    class_mapping = normalize_class_mapping(metadata)

    input_scaled = scaler.transform(input_df[features])
    pred_encoded = int(model.predict(input_scaled)[0])
    pred_label = class_mapping.get(pred_encoded, str(pred_encoded))

    result = {
        "encoded_prediction": pred_encoded,
        "label": pred_label,
        "confidence": None,
        "probability_table": None,
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        model_classes = [int(cls) for cls in model.classes_]

        rows = []
        for cls, probability in zip(model_classes, proba):
            label = class_mapping.get(int(cls), str(cls))
            rows.append(
                {
                    "Category": label,
                    "Probability": float(probability),
                    "Probability (%)": float(probability * 100),
                    "Color": AQI_COLORS.get(label, "#64748b"),
                }
            )

        prob_df = pd.DataFrame(rows)
        prob_df["Order"] = prob_df["Category"].apply(lambda x: AQI_ORDER.index(x) if x in AQI_ORDER else 99)
        prob_df = prob_df.sort_values("Order").drop(columns="Order")

        result["confidence"] = float(np.max(proba))
        result["probability_table"] = prob_df

    return result


def make_probability_chart(prob_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=prob_df["Probability (%)"],
                y=prob_df["Category"],
                orientation="h",
                marker=dict(color=prob_df["Color"]),
                text=prob_df["Probability (%)"].round(2).astype(str) + "%",
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Probabilitas: %{x:.2f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=20, t=20, b=10),
        xaxis_title="Probabilitas (%)",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        xaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,0.25)"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def make_pollutant_chart(input_df: pd.DataFrame) -> go.Figure:
    values = input_df.iloc[0][FEATURES].astype(float)
    fig = go.Figure(
        data=[
            go.Bar(
                x=values.index,
                y=values.values,
                marker=dict(color=["#38bdf8", "#818cf8", "#2dd4bf", "#fb923c"]),
                text=[f"{v:.0f}" for v in values.values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Nilai AQI: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=20, t=25, b=10),
        yaxis_title="Nilai AQI",
        xaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
    )
    return fig


def risk_score_from_category(category: str) -> int:
    if category not in AQI_ORDER:
        return 0
    return int((AQI_ORDER.index(category) + 1) / len(AQI_ORDER) * 100)


def rule_based_reference(input_df: pd.DataFrame) -> str:
    """
    Referensi sederhana berdasarkan nilai polutan tertinggi.
    Ini bukan pengganti model, hanya penjelasan tambahan agar user awam lebih memahami input.
    """
    max_value = float(input_df[FEATURES].iloc[0].max())
    if max_value <= 50:
        return "Good"
    if max_value <= 100:
        return "Moderate"
    if max_value <= 150:
        return "Unhealthy for Sensitive Groups"
    if max_value <= 200:
        return "Unhealthy"
    if max_value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def create_example_csv() -> bytes:
    example = pd.DataFrame(
        [
            {"CO AQI Value": 1, "Ozone AQI Value": 35, "NO2 AQI Value": 12, "PM2.5 AQI Value": 42},
            {"CO AQI Value": 3, "Ozone AQI Value": 95, "NO2 AQI Value": 28, "PM2.5 AQI Value": 115},
            {"CO AQI Value": 8, "Ozone AQI Value": 175, "NO2 AQI Value": 80, "PM2.5 AQI Value": 210},
        ]
    )
    return example.to_csv(index=False).encode("utf-8")


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### 🌫️ Air Quality App")
    st.caption("Dashboard prediksi kategori kualitas udara berbasis model machine learning.")

    st.markdown("#### File model yang dibutuhkan")
    st.code(
        "model/\n"
        "├── best_model_no_tuning.pkl\n"
        "├── scaler_no_tuning.pkl\n"
        "└── model_metadata_no_tuning.json",
        language="text",
    )

    st.markdown("#### Fitur input")
    for feat in FEATURES:
        st.markdown(f"<span class='feature-pill'>{feat}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.download_button(
        label="⬇️ Download contoh CSV",
        data=create_example_csv(),
        file_name="sample_input.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# HERO
# =========================================================
st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">🌫️ {APP_TITLE}</div>
        <div class="hero-subtitle">
            {APP_SUBTITLE}. Aplikasi ini membantu pengguna memahami kategori kualitas udara dari nilai CO, Ozone,
            NO2, dan PM2.5 dengan tampilan yang rapi, interaktif, dan mudah dipahami.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================
try:
    model, scaler, metadata = load_artifacts()
    model_ready = True
except Exception as error:
    model_ready = False
    metadata = {}

    st.error("Model belum bisa dimuat.")
    st.markdown(
        """
        Aplikasi ini sudah siap, tetapi file model belum ditemukan. Jalankan salah satu cara berikut:

        **Cara paling mudah**
        1. Jalankan notebook final sampai bagian export model.
        2. Ambil 3 file dari folder `outputs_no_tuning/`.
        3. Masukkan ke folder `model/` pada project deployment ini.
        4. Jalankan ulang aplikasi dengan `streamlit run app.py`.

        **Cara alternatif**
        Jalankan script `export_model_from_csv.py` jika kamu punya file `global_air_pollution_dataset.csv`.
        """
    )
    st.code(str(error), language="text")
    st.stop()


# =========================================================
# MODEL SUMMARY
# =========================================================
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">Model Final</div>
            <div class="big-number">{metadata.get("model_name", "Model")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    accuracy = metadata.get("accuracy")
    st.metric("Accuracy", f"{accuracy * 100:.2f}%" if isinstance(accuracy, (int, float)) else "-")

with col_c:
    weighted_f1 = metadata.get("weighted_f1")
    st.metric("Weighted F1", f"{weighted_f1 * 100:.2f}%" if isinstance(weighted_f1, (int, float)) else "-")

with col_d:
    st.metric("Tuning", "Tidak digunakan" if not metadata.get("hyperparameter_tuning", False) else "Digunakan")


# =========================================================
# TABS
# =========================================================
tab_predict, tab_batch, tab_model, tab_guide = st.tabs(
    ["🔮 Prediksi Manual", "📦 Batch CSV", "🧠 Tentang Model", "📘 Panduan"]
)


# =========================================================
# TAB 1: MANUAL PREDICTION
# =========================================================
with tab_predict:
    left, right = st.columns([0.92, 1.08], gap="large")

    with left:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader("Masukkan Nilai Polutan")

        preset = st.selectbox(
            "Pilih contoh kondisi",
            [
                "Custom",
                "Udara Baik",
                "Sedang",
                "Tidak Sehat untuk Kelompok Sensitif",
                "Tidak Sehat",
                "Sangat Tidak Sehat",
                "Berbahaya",
            ],
        )

        presets = {
            "Udara Baik": (1, 35, 12, 42),
            "Sedang": (2, 80, 25, 85),
            "Tidak Sehat untuk Kelompok Sensitif": (4, 125, 45, 135),
            "Tidak Sehat": (6, 170, 70, 180),
            "Sangat Tidak Sehat": (9, 240, 120, 260),
            "Berbahaya": (12, 360, 180, 410),
        }

        default_values = presets.get(preset, (1, 35, 12, 42))

        co_value = st.slider("CO AQI Value", 0, 500, int(default_values[0]), help="Nilai indeks AQI untuk polutan CO.")
        ozone_value = st.slider("Ozone AQI Value", 0, 500, int(default_values[1]), help="Nilai indeks AQI untuk polutan Ozone.")
        no2_value = st.slider("NO2 AQI Value", 0, 500, int(default_values[2]), help="Nilai indeks AQI untuk polutan NO2.")
        pm25_value = st.slider("PM2.5 AQI Value", 0, 500, int(default_values[3]), help="Nilai indeks AQI untuk polutan PM2.5.")

        input_df = build_input_dataframe(co_value, ozone_value, no2_value, pm25_value)
        st.markdown("</div>", unsafe_allow_html=True)

        st.plotly_chart(make_pollutant_chart(input_df), use_container_width=True)

    with right:
        prediction = predict_category(model, scaler, metadata, input_df)
        label = prediction["label"]
        color = AQI_COLORS.get(label, "#0f172a")
        advice = AQI_ADVICE.get(
            label,
            {
                "status": "Kategori belum dikenali.",
                "advice": "Periksa kembali input dan metadata model.",
                "risk": "-",
            },
        )
        confidence = prediction["confidence"]
        confidence_text = f"{confidence * 100:.2f}%" if confidence is not None else "Tidak tersedia"

        st.markdown(
            f"""
            <div class="prediction-card" style="background: linear-gradient(135deg, {color}, #0f172a);">
                <div style="font-size:0.86rem; opacity:0.9; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">
                    Hasil Prediksi Model
                </div>
                <div style="font-size:2.55rem; font-weight:850; margin-top:0.25rem; letter-spacing:-0.05em;">
                    {label}
                </div>
                <div style="font-size:1rem; line-height:1.65; margin-top:0.6rem; opacity:0.96;">
                    {advice["status"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", confidence_text)
        c2.metric("Risk Level", advice["risk"])
        c3.metric("Rule Reference", rule_based_reference(input_df))

        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader("Rekomendasi Singkat")
        st.write(advice["advice"])

        risk_score = risk_score_from_category(label)
        st.progress(risk_score / 100)
        st.caption(f"Skor risiko visual: {risk_score}/100 berdasarkan kategori prediksi.")
        st.markdown("</div>", unsafe_allow_html=True)

        if prediction["probability_table"] is not None:
            st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
            st.subheader("Probabilitas Setiap Kategori")
            st.plotly_chart(make_probability_chart(prediction["probability_table"]), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 2: BATCH CSV
# =========================================================
with tab_batch:
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.subheader("Prediksi Banyak Data Menggunakan CSV")
    st.write(
        "Upload file CSV yang memiliki 4 kolom fitur: "
        "`CO AQI Value`, `Ozone AQI Value`, `NO2 AQI Value`, dan `PM2.5 AQI Value`."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_cols = [col for col in FEATURES if col not in batch_df.columns]

            if missing_cols:
                st.error(f"Kolom berikut belum ada di CSV: {missing_cols}")
            else:
                features = metadata.get("features", FEATURES)
                if isinstance(features, str):
                    features = [item.strip() for item in features.split(",")]

                scaled_batch = scaler.transform(batch_df[features])
                pred_encoded = model.predict(scaled_batch)
                class_mapping = normalize_class_mapping(metadata)

                result_df = batch_df.copy()
                result_df["Predicted AQI Category"] = [class_mapping.get(int(pred), str(pred)) for pred in pred_encoded]

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(scaled_batch)
                    result_df["Prediction Confidence"] = np.max(proba, axis=1)

                st.success("Prediksi batch berhasil dibuat.")
                st.dataframe(result_df, use_container_width=True)

                st.download_button(
                    "⬇️ Download hasil prediksi",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="aqi_batch_prediction_result.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                summary = result_df["Predicted AQI Category"].value_counts().reset_index()
                summary.columns = ["Category", "Count"]
                summary["Color"] = summary["Category"].map(AQI_COLORS).fillna("#64748b")

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=summary["Category"],
                            y=summary["Count"],
                            marker=dict(color=summary["Color"]),
                            text=summary["Count"],
                            textposition="outside",
                        )
                    ]
                )
                fig.update_layout(
                    title="Distribusi Hasil Prediksi Batch",
                    height=360,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=20, t=55, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as batch_error:
            st.error("File CSV belum berhasil diproses.")
            st.code(str(batch_error), language="text")
    else:
        st.info("Belum ada file CSV yang diupload. Kamu bisa download contoh CSV dari sidebar.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 3: MODEL INFORMATION
# =========================================================
with tab_model:
    left_info, right_info = st.columns([1, 1], gap="large")

    with left_info:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader("Ringkasan Model")
        model_info = {
            "Model final": metadata.get("model_name", "-"),
            "Target": metadata.get("target", "AQI Category"),
            "Hyperparameter tuning": "Tidak digunakan" if not metadata.get("hyperparameter_tuning", False) else "Digunakan",
            "SMOTE": "Digunakan" if metadata.get("smote_used", False) else "Tidak digunakan",
            "Accuracy": f"{metadata.get('accuracy', 0) * 100:.2f}%" if isinstance(metadata.get("accuracy"), (int, float)) else "-",
            "Weighted F1": f"{metadata.get('weighted_f1', 0) * 100:.2f}%" if isinstance(metadata.get("weighted_f1"), (int, float)) else "-",
            "Macro F1": f"{metadata.get('macro_f1', 0) * 100:.2f}%" if isinstance(metadata.get("macro_f1"), (int, float)) else "-",
        }
        st.table(pd.DataFrame(model_info.items(), columns=["Item", "Keterangan"]))
        st.markdown("</div>", unsafe_allow_html=True)

    with right_info:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.subheader("Kenapa Deployment Pakai 1 Model?")
        st.write(
            "Aplikasi deployment sebaiknya memakai satu model final saja agar alur prediksi lebih jelas, stabil, "
            "dan mudah dijelaskan saat presentasi. Model lain cukup ditempatkan di notebook sebagai proses seleksi, "
            "sedangkan aplikasi hanya menampilkan model terbaik yang sudah dipilih."
        )
        st.write(
            "Dengan pendekatan ini, aplikasi tidak terlihat seperti eksperimen perbandingan model, tetapi sebagai "
            "produk akhir dari proses klasifikasi kualitas udara."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.subheader("Fitur yang Digunakan")
    feature_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Explanation": [
                "Nilai AQI untuk polutan karbon monoksida.",
                "Nilai AQI untuk polutan ozone.",
                "Nilai AQI untuk polutan nitrogen dioksida.",
                "Nilai AQI untuk polutan partikel halus PM2.5.",
            ],
        }
    )
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    st.caption(
        "Catatan: AQI Value tidak digunakan sebagai input model untuk menghindari data leakage. "
        "Aplikasi hanya memakai nilai AQI dari masing-masing polutan utama."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 4: GUIDE
# =========================================================
with tab_guide:
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.subheader("Cara Menjalankan Aplikasi")
    st.markdown(
        """
        **1. Jalankan notebook final sampai bagian export model**  
        Pastikan folder `outputs_no_tuning/` berisi:
        - `best_model_no_tuning.pkl`
        - `scaler_no_tuning.pkl`
        - `model_metadata_no_tuning.json`

        **2. Copy tiga file tersebut ke folder `model/` pada project deployment**  
        Struktur akhirnya:
        ```text
        aqi_streamlit_deployment/
        ├── app.py
        ├── requirements.txt
        └── model/
            ├── best_model_no_tuning.pkl
            ├── scaler_no_tuning.pkl
            └── model_metadata_no_tuning.json
        ```

        **3. Install library dan jalankan Streamlit**
        ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```

        **4. Untuk deploy online**
        Upload project ke GitHub, lalu deploy melalui Streamlit Community Cloud atau platform lain.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<p class='footer-note'>Aplikasi ini dibuat untuk kebutuhan edukasi dan demonstrasi model klasifikasi kualitas udara. Hasil prediksi sebaiknya tidak digunakan sebagai satu-satunya dasar keputusan kesehatan atau kebijakan publik.</p>", unsafe_allow_html=True)
