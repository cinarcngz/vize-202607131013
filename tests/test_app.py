
import pandas as pd

from app import detect_anomalies


def test_detect_anomalies():

    data = pd.DataFrame({
        "sensor": [
            "trafik",
            "trafik",
            "trafik",
            "trafik"
        ],
        "deger": [10, 11, 12, 100]
    })

    result = detect_anomalies(data)

    assert len(result) == 4
    assert result["anomali"].iloc[-1] == True


def test_no_anomalies():

    data = pd.DataFrame({
        "sensor": [
            "trafik",
            "trafik",
            "trafik",
            "trafik"
        ],
        "deger": [10, 11, 10, 11]
    })

    result = detect_anomalies(data)

    assert result["anomali"].sum() == 0


def test_empty_data():

    data = pd.DataFrame()

    result = detect_anomalies(data)

    assert result.empty
