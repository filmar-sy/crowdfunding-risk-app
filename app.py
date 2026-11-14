import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_crowdfunding.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Crowdfunding Risk Predictor",
    page_icon="📊",
    layout="centered",
)

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata

try:
    model, metadata = load_assets()
except Exception as exc:
    st.error("Model belum dapat dimuat. Periksa file model dan metadata di folder aplikasi.")
    st.exception(exc)
    st.stop()

FEATURES = metadata["features"]
THRESHOLD = float(metadata["failure_threshold"])
CATEGORIES = metadata["supported_categories"]
COUNTRIES = metadata["supported_countries"]

st.title("Prediksi Risiko Kampanye Crowdfunding")
st.write(
    "Aplikasi ini membantu calon kreator meninjau risiko kampanye Kickstarter "
    "sebelum diluncurkan berdasarkan pola pada data historis."
)

with st.expander("Penting: cara membaca hasil"):
    st.write(
        "Hasil prediksi adalah peringatan berbasis model, bukan kepastian bahwa "
        "kampanye akan gagal atau berhasil. Model dirancang untuk menangkap "
        "sebanyak mungkin kampanye gagal, sehingga kampanye yang akhirnya sukses "
        "juga dapat menerima peringatan."
    )

st.header("Masukkan rencana kampanye")
with st.form("campaign_form"):
    start_date = st.date_input(
        "Tanggal mulai kampanye",
        value=date.today(),
        help="Masukkan tanggal mulai yang direncanakan."
    )
    end_date = st.date_input(
        "Tanggal akhir kampanye",
        value=date.today(),
        help="Masukkan tanggal akhir yang direncanakan."
    )
    category = st.selectbox("Kategori kampanye", options=CATEGORIES)
    country = st.selectbox("Negara kreator", options=COUNTRIES)
    images = st.radio(
        "Apakah halaman kampanye memiliki gambar?",
        options=["Ya", "Tidak"],
        horizontal=True
    )
    rewards = st.number_input(
        "Jumlah reward",
        min_value=1,
        max_value=999,
        value=5,
        step=1,
        help="Masukkan jumlah tingkatan reward yang direncanakan."
    )
    submitted = st.form_submit_button("Prediksi risiko kampanye", type="primary")

if submitted:
    duration = (end_date - start_date).days
    if end_date <= start_date:
        st.error("Tanggal akhir harus lebih akhir daripada tanggal mulai.")
    elif duration < 1:
        st.error("Durasi kampanye harus minimal satu hari.")
    else:
        input_row = pd.DataFrame([{
            "Duration": duration,
            "Category_Kickstarter": category,
            "Creator_Country": country,
            "Images_Bool": 1 if images == "Ya" else 0,
            "Rewards": int(rewards),
        }], columns=FEATURES)

        class_labels = list(model.named_steps["model"].classes_)
        failure_index = class_labels.index(0)
        failure_score = float(model.predict_proba(input_row)[0, failure_index])

        st.divider()
        st.header("Hasil prediksi")
        st.metric("Durasi kampanye yang dihitung aplikasi", f"{duration} hari")
        st.metric("Skor estimasi risiko kegagalan", f"{failure_score:.1%}")
        st.caption(
            "Skor ini merupakan keluaran model dan belum boleh dianggap sebagai "
            "probabilitas aktual yang telah dikalibrasi."
        )

        if failure_score >= THRESHOLD:
            st.error("Peringatan risiko kegagalan")
            st.write(
                "Model menemukan pola yang berkaitan dengan risiko kegagalan. "
                "Pertimbangkan untuk meninjau kembali rancangan kampanye sebelum diluncurkan."
            )
        else:
            st.success("Risiko kegagalan lebih rendah menurut model")
            st.write(
                "Skor berada di bawah batas peringatan model. Ini bukan jaminan "
                "bahwa kampanye akan berhasil."
            )

        with st.expander("Ringkasan data yang dianalisis"):
            st.dataframe(input_row, use_container_width=True, hide_index=True)

st.divider()
st.header("Tentang model")
st.write(
    "Model menggunakan algoritma Extra Trees dengan lima fitur: durasi kampanye, "
    "kategori Kickstarter, negara kreator, keberadaan gambar, dan jumlah reward."
)
st.write(f"Batas peringatan risiko kegagalan: **{THRESHOLD:.3f}**.")
st.write(
    "Pada evaluasi test proyek ini, Recall kampanye gagal adalah sekitar 90,74%, "
    "sedangkan Precision kampanye gagal sekitar 16,39%. Artinya model berhasil "
    "menandai banyak kampanye gagal, tetapi banyak kampanye sukses juga menerima "
    "peringatan. Kinerja pada data atau periode lain dapat berbeda."
)
st.caption(
    "Sumber model: data historis Kickstarter dalam proyek ini. Aplikasi tidak "
    "menjamin hasil pendanaan dan bukan pengganti evaluasi strategi oleh kreator."
)
