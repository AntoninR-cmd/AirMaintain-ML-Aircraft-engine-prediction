import streamlit as st
import requests
import pandas as pd
import os


SENSOR_COLUMNS = [
    f"MesureCapteur{i:02d}"
    for i in range(1, 22)
]

COLUMNS = [
    "IdMoteur",
    "Cycle",
    "ParameterOpe1",
    "ParameterOpe2",
    "ParameterOpe3",
    *SENSOR_COLUMNS
]

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


API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)

st.set_page_config(
    page_title="AeroMaintain",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ AeroMaintain")
st.caption(
    "Prédiction de durée de vie restante "
    "pour moteurs aéronautiques"
)

def get_api_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )
        return response.status_code == 200

    except requests.RequestException:
        return False


@st.cache_data(ttl=30)
def get_model_info():
    try:
        response = requests.get(
            f"{API_URL}/model/info",
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()

    except requests.RequestException:
        pass

    return None

if get_api_health():
    st.success("API connectée")
else:
    st.error(
        "API indisponible. Vérifiez les conteneurs Docker."
    )

model_info = get_model_info()

if model_info is not None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Modèle RUL",
            model_info["rul_model"],
        )

    with col2:
        st.metric(
            "Features",
            model_info["feature_count"],
        )

    with col3:
        st.metric(
            "Fenêtres temporelles",
            ", ".join(
                str(x)
                for x in model_info["windows"]
            ),
        )

    with st.expander(
        "Modèles de quantiles"
    ):
        st.write(
            f"**q10 :** {model_info['q10_model']}"
        )
        st.write(
            f"**q50 :** {model_info['q50_model']}"
        )
        st.write(
            f"**q90 :** {model_info['q90_model']}"
        )

st.divider()
st.header("Analyse d'un moteur")

uploaded_file = st.file_uploader(
    "Historique C-MAPSS",
    type=["txt"],
    help=(
        "Sélectionnez un fichier contenant "
        "l'historique des cycles moteur."
    ),
)
if uploaded_file is not None:
    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    with st.expander("Aperçu des données"):
        st.dataframe(
            df.head(10),
            use_container_width=True,
        )

    engine_ids = df["IdMoteur"].unique()

    col1, col2 = st.columns(2)

    with col1:
        selected_engine = st.selectbox(
            "Moteur",
            engine_ids,
        )

    with col2:
        fd = st.selectbox(
            "Dataset",
            ["001", "002", "003", "004"],
            format_func=lambda x: f"FD{x}",
        )

    engine_df = df[
        df["IdMoteur"] == selected_engine
    ].copy()

    if st.button(
        "Analyser le moteur",
        type="primary",
        use_container_width=True,
    ):
        payload = dataframe_to_payload(
            engine_df,
            fd,
            selected_engine
        )

        try:
            with st.spinner(
                "Analyse du moteur en cours..."
            ):
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
                        "RUL modèle",
                        f"{prediction['rul']:.1f} cycles",
                    )

                with col2:
                    st.metric(
                        "Médiane q50",
                        f"{prediction['q50']:.1f} cycles",
                    )

                with col3:
                    st.metric(
                        "Quantile q10",
                        f"{prediction['q10']:.1f} cycles",
                    )

                with col4:
                    st.metric(
                        "Quantile q90",
                        f"{prediction['q90']:.1f} cycles",
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
                with st.expander(
                    "Détails techniques"
                ):
                    st.code(response.text)
        except requests.RequestException:
            st.error("Impossible de joindre l'API")