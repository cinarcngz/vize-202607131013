
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa ayarlari
st.set_page_config(
    page_title="Akilli Sehir Asistani",
    page_icon="🏙️",
    layout="wide"
)

# Dosya yolları
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "sehir_verileri.csv"
DOC_PATH = BASE_DIR / "docs" / "sehir_bilgisi.md"

st.title("🏙️ Akilli Sehir Yonetim Sistemi")
st.write("RAG Bilgi Asistani ve Anomali Tespiti")

# Bilgi dokumanini oku
def load_knowledge():
    if DOC_PATH.exists():
        return DOC_PATH.read_text(encoding="utf-8")
    return ""

# RAG bilgi asistanı
def rag_answer(question, document):
    if not document.strip():
        return "Bilgi dokumani bulunamadi."

    chunks = [
        part.strip()
        for part in document.split("\n\n")
        if len(part.strip()) > 20
    ]

    if not chunks:
        return "Dokumanda ilgili bilgi bulunamadi."

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks + [question])

    scores = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    ).flatten()

    best_index = scores.argmax()

    if scores[best_index] == 0:
        return "Bu konuda bilgi dokumaninda eslesen bilgi bulunamadi."

    return chunks[best_index]

# Sensör verilerini oku
def load_sensor_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

# Anomali tespiti
def detect_anomalies(data):
    if data.empty:
        return pd.DataFrame()

    results = []

    for sensor in data["sensor"].unique():
        group = data[data["sensor"] == sensor].copy()

        q1 = group["deger"].quantile(0.25)
        q3 = group["deger"].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        group["anomali"] = (
            (group["deger"] < lower) |
            (group["deger"] > upper)
        )

        results.append(group)

    return pd.concat(results, ignore_index=True)

# Sekmeler
tab1, tab2, tab3 = st.tabs([
    "💬 RAG Bilgi Asistani",
    "📊 Anomali Tespiti",
    "📁 Sensor Verileri"
])

# RAG sekmesi
with tab1:
    st.header("Sehir Bilgi Asistani")

    question = st.text_input(
        "Akilli sehir hakkinda bir soru sorun:"
    )

    if st.button("Cevapla"):
        document = load_knowledge()

        if question.strip():
            answer = rag_answer(question, document)
            st.success(answer)
        else:
            st.warning("Lutfen bir soru yazin.")

# Anomali sekmesi
with tab2:
    st.header("Sensor Anomali Analizi")

    data = load_sensor_data()

    if not data.empty:
        analyzed = detect_anomalies(data)

        st.subheader("Tespit Edilen Anomaliler")

        anomalies = analyzed[analyzed["anomali"]]

        if not anomalies.empty:
            st.warning(
                f"{len(anomalies)} anomali tespit edildi."
            )
            st.dataframe(anomalies)
        else:
            st.success("Anomali tespit edilmedi.")

        st.subheader("Sensor Degerleri")
        st.bar_chart(
            data,
            x="sensor",
            y="deger"
        )
    else:
        st.error("Sensor veri dosyasi bulunamadi.")

# Veri sekmesi
with tab3:
    st.header("Sehir Sensor Verileri")

    data = load_sensor_data()

    if not data.empty:
        st.dataframe(data, use_container_width=True)
    else:
        st.warning("Veri bulunamadi.")
