
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "sehir_verileri.csv"
DOC_PATH = BASE_DIR / "docs" / "sehir_bilgisi.md"


def load_knowledge():
    if not DOC_PATH.exists():
        return []

    document = DOC_PATH.read_text(encoding="utf-8")

    return [
        part.strip()
        for part in document.split("\n\n")
        if len(part.strip()) > 20
    ]


def rag_answer(question):
    chunks = load_knowledge()

    if not chunks:
        return "Bilgi dokumani bulunamadi."

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks + [question])

    scores = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    ).flatten()

    best_index = scores.argmax()

    if scores[best_index] == 0:
        return "Soruyla ilgili bilgi bulunamadi."

    return chunks[best_index]


def load_sensor_data():
    if not DATA_PATH.exists():
        print("Sensor veri dosyasi bulunamadi.")
        return pd.DataFrame()

    return pd.read_csv(DATA_PATH)


def detect_anomalies(data):
    if data.empty:
        return pd.DataFrame()

    results = []

    for sensor in data["sensor"].unique():

        group = data[
            data["sensor"] == sensor
        ].copy()

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


def show_sensor_data():
    data = load_sensor_data()

    if not data.empty:
        print("\n--- SENSOR VERILERI ---")
        print(data.to_string(index=False))


def show_anomalies():
    data = load_sensor_data()

    if data.empty:
        return

    analyzed = detect_anomalies(data)

    anomalies = analyzed[
        analyzed["anomali"]
    ]

    print("\n--- ANOMALI SONUCLARI ---")

    if anomalies.empty:
        print("Anomali tespit edilmedi.")
    else:
        print(anomalies.to_string(index=False))
        print(
            "\nToplam anomali:",
            len(anomalies)
        )


def main():
    while True:

        print("\n==============================")
        print(" AKILLI SEHIR YONETIM SISTEMI")
        print("==============================")
        print("1 - RAG Bilgi Asistani")
        print("2 - Anomali Tespiti")
        print("3 - Sensor Verilerini Goruntule")
        print("0 - Cikis")

        choice = input("\nSeciminiz: ")

        if choice == "1":
            question = input(
                "\nSehir hakkinda sorunuzu yazin: "
            )

            print("\nAsistan cevabi:")
            print(rag_answer(question))

        elif choice == "2":
            show_anomalies()

        elif choice == "3":
            show_sensor_data()

        elif choice == "0":
            print("Programdan cikiliyor.")
            break

        else:
            print("Gecersiz secim!")


if __name__ == "__main__":
    main()
