
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "standalone"

st.set_page_config(
    page_title="AttriPredict AI",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        MODEL_DIR / "AttriPredict_ANN_Model.keras"
    )

    encoder = joblib.load(
        MODEL_DIR / "AttriPredict_Encoder.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "AttriPredict_Scaler.pkl"
    )

    feature_columns = joblib.load(
        MODEL_DIR / "AttriPredict_Feature_Columns.pkl"
    )

    return model, encoder, scaler, feature_columns


st.title("📊 AttriPredict AI")
st.subheader("Employee Attrition Prediction System")

st.write(
    "An Artificial Neural Network (ANN) based system "
    "that estimates employee attrition risk."
)

try:
    model, encoder, scaler, feature_columns = load_artifacts()
    st.success("ANN Model and preprocessing artifacts loaded successfully!")

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


st.info(
    "Enter employee details below to estimate "
    "the likelihood of employee attrition."
)

numerical_cols = feature_columns["numerical_columns"]
categorical_cols = feature_columns["categorical_columns"]

with st.form("employee_form"):

    st.subheader("Employee Details")

    numeric_values = {}
    categorical_values = {}

    st.markdown("### Numerical Information")

    numeric_layout = st.columns(3)

    for i, col in enumerate(numerical_cols):
        with numeric_layout[i % 3]:
            numeric_values[col] = st.number_input(
                col,
                value=0.0,
                min_value=0.0,
                step=1.0
            )

    st.markdown("### Categorical Information")

    categorical_layout = st.columns(2)

    for i, col in enumerate(categorical_cols):
        with categorical_layout[i % 2]:
            options = list(encoder.categories_[i])

            categorical_values[col] = st.selectbox(
                col,
                options=options
            )

    submitted = st.form_submit_button(
        "Predict Employee Attrition",
        use_container_width=True
    )


if submitted:

    try:
        numeric_df = pd.DataFrame(
            [[numeric_values[col] for col in numerical_cols]],
            columns=numerical_cols
        )

        categorical_df = pd.DataFrame(
            [[categorical_values[col] for col in categorical_cols]],
            columns=categorical_cols
        )

        scaled_numeric = scaler.transform(numeric_df)
        encoded_categorical = encoder.transform(categorical_df)

        model_input = np.concatenate(
            [scaled_numeric, encoded_categorical],
            axis=1
        )

        probability = float(
            model.predict(model_input, verbose=0)[0][0]
        )

        st.divider()
        st.subheader("Prediction Result")

        if probability >= 0.5:
            st.error("Prediction: Employee May Leave")
        else:
            st.success("Prediction: Employee Likely to Stay")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Leave Probability",
                f"{probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Stay Probability",
                f"{(1 - probability) * 100:.2f}%"
            )

        st.progress(probability)

        st.caption(
            "This is an experimental model prediction, "
            "not a definitive assessment of an employee."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
