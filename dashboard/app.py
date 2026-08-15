import streamlit as st
import requests
import pandas as pd
from aeromaintain.config import COLUMNS


def dataframe_to_payload(
    df,
    fd,
    engine_id
):
    history = []

    for index, row in df.iterrows():
        reading = {
            "cycle": int(row["Cycle"]),
            "parameter_ope1": float(row["ParameterOpe1"]),
            "parameter_ope2": float(row["ParameterOpe2"]),
            "parameter_ope3": float(row["ParameterOpe3"])
        }

        for i in range(1, 22):
            reading[f"sensor_{i:02d}"] = float(
                row[f"MesureCapteur{i:02d}"]
            )

        history.append(reading)

    payload = {
        "fd": fd,
        "engine_id": int(engine_id),
        "history": history
    }

    return payload


API_URL = "http://localhost:8000"

st.title("AeroMaintain")

if st.button("Testez l'API"):
    try:
        response = requests.get(
            f"{API_URL}/health"
        )

        if response.status_code == 200:
            st.success("L'API est lancée")
        else:
            st.error("L'API a échoué")
    except requests.RequestException:
        st.error("Impossible de joindre l'API")

st.header("Modèle")

if st.button("Informations modèle"):
    try:
        response = requests.get(
            f"{API_URL}/model/info",
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()

            st.success("Informations du modèle récupérées")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Modèle RUL",
                    data["rul_model"]
                )

            with col2:
                st.metric(
                    "Features",
                    data["feature_count"]
                )

            with col3:
                st.metric(
                    "Fenêtres",
                    ", ".join(
                        str(x)
                        for x in data["windows"]
                    )
                )

            col4, col5, col6 = st.columns(3)

            with col4:
                st.metric(
                    "Quantile 10 %",
                    data["q10_model"]
                )

            with col5:
                st.metric(
                    "Quantile 50 %",
                    data["q50_model"]
                )

            with col6:
                st.metric(
                    "Quantile 90 %",
                    data["q90_model"]
                )

        else:
            st.error(
                f"Erreur API : {response.status_code}"
            )

    except requests.RequestException:
        st.error(
            "Impossible de joindre l'API"
        )

st.header("Prediction RUL")

uploaded_file = st.file_uploader(
    "Importer l'historique d'un moteur",
    type=["txt"]
)
if uploaded_file is not None:
    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    st.dataframe(df.head())

    engine_ids = df["IdMoteur"].unique()

    selected_engine = st.selectbox(
        "Moteur",
        engine_ids
    )

    engine_df = df[
        df["IdMoteur"] == selected_engine
    ].copy()

    fd = st.selectbox(
        "Dataset FD",
        ["001", "002", "003", "004"]
    )

    if st.button("Prédire la RUL"):
        payload = dataframe_to_payload(
            engine_df,
            fd,
            selected_engine
        )

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                st.success("Prédiction réussie")

                prediction = response.json()

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "RUL prédite",
                        prediction["rul"]
                    )

                with col2:
                    st.metric(
                        "médiane",
                        prediction["q50"]
                    )

                with col3:
                    st.metric(
                        "Borne basse",
                        prediction["q10"]
                    )

                with col4:
                    st.metric(
                        "Borne haute",
                        prediction["q90"]
                    )

                st.info(
                    f"Le moteur {prediction['engine_id']} "
                    f"est au cycle {prediction['cycle']}."
                )

                st.write(
                    f"La durée de vie estimée du moteur est "
                    f"de {prediction['rul']} cycles."
                )
            else:
                st.error(f"Erreur API {response.status_code}")
                st.code(response.text)
        except requests.RequestException:
            st.error("Impossible de joindre l'API")