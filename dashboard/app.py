import streamlit as st
import requests


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

            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric(
                    "Modèle RUL",
                    data["rul_model"]
                )

            with col2:
                st.metric(
                    "Modèle q10",
                    data["q10_model"]
                )

            with col3:
                st.metric(
                    "Modèle q50",
                    data["q50_model"]
                )

            with col4:
                st.metric(
                    "Modèle q90",
                    data["q90_model"]
                )

            with col5:
                st.metric(
                    "Features",
                    data["feature_count"]
                )

            with col6:
                st.metric(
                    "Fenêtres",
                    ", ".join(
                        str(x)
                        for x in data["windows"]
                    )
                )

        else:
            st.error(
                f"Erreur API : {response.status_code}"
            )

    except requests.RequestException:
        st.error(
            "Impossible de joindre l'API"
        )