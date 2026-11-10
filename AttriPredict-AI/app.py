
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AttriPredict AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "standalone"


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background: #f4f7fc;
}

[data-testid="stSidebar"] {
    background: #132342;
}

[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    background: linear-gradient(120deg, #14264b, #2864bd);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 20px;
}

.hero h1 {
    color: white;
    font-size: 32px;
}

.hero p {
    color: #e5edff;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #e4eaf4;
    box-shadow: 0 2px 8px rgba(20,40,80,.04);
}

div.stButton > button,
div.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

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


try:
    model, encoder, scaler, feature_columns = load_artifacts()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


numerical_cols = feature_columns["numerical_columns"]
categorical_cols = feature_columns["categorical_columns"]


# ---------------- SESSION HISTORY ----------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ---------------- PREDICTION FUNCTION ----------------

def make_prediction(numeric_values, categorical_values):

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

    probability = float(np.clip(probability, 0, 1))

    prediction = (
        "May Leave" if probability >= 0.5
        else "Likely to Stay"
    )

    if probability < 0.30:
        risk = "Low"
    elif probability < 0.60:
        risk = "Moderate"
    else:
        risk = "High"

    return probability, prediction, risk


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("## 📊 AttriPredict AI")
    st.caption("HR Analytics Platform")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Predict Attrition",
            "History & Analytics",
            "About"
        ]
    )

    st.divider()

    st.caption("Powered by Artificial Neural Network")
    st.caption("Built with Python and Streamlit")


# =================================================
# OVERVIEW PAGE
# =================================================

if page == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>Employee Attrition Intelligence</h1>
        <p>AI-powered employee attrition prediction and HR analytics.</p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.prediction_history

    total = len(history)

    may_leave = sum(
        x["Prediction"] == "May Leave"
        for x in history
    )

    likely_stay = total - may_leave

    st.markdown("### Dashboard Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Predictions", total)
    c2.metric("Predicted May Leave", may_leave)
    c3.metric("Predicted Likely to Stay", likely_stay)

    st.markdown("### Platform Features")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **🔍 Attrition Prediction**

        Enter employee information and estimate attrition probability
        using the trained ANN model.
        """)

    with col2:
        st.info("""
        **📈 HR Analytics**

        Review prediction history, explore charts and download reports.
        """)

    st.markdown("### Model Information")

    a, b, c = st.columns(3)

    a.metric("Model Type", "ANN")
    b.metric(
        "Input Features",
        len(numerical_cols) + len(categorical_cols)
    )
    c.metric("Output Type", "Binary Classification")

    st.caption(
        "This application is intended for educational and analytical "
        "demonstration, not as the sole basis for employment decisions."
    )


# =================================================
# PREDICTION PAGE
# =================================================

elif page == "Predict Attrition":

    st.title("🔍 Employee Attrition Prediction")

    st.write(
        "Enter employee information to estimate attrition likelihood."
    )

    numeric_values = {}
    categorical_values = {}

    with st.form("employee_form"):

        st.markdown("### 1. Numerical Information")

        numeric_layout = st.columns(3)

        for i, col in enumerate(numerical_cols):

            with numeric_layout[i % 3]:

                numeric_values[col] = st.number_input(
                    col,
                    value=0.0,
                    min_value=0.0,
                    step=1.0
                )

        st.divider()

        st.markdown("### 2. Categorical Information")

        categorical_layout = st.columns(2)

        for i, col in enumerate(categorical_cols):

            with categorical_layout[i % 2]:

                options = list(encoder.categories_[i])

                categorical_values[col] = st.selectbox(
                    col,
                    options=options
                )

        submitted = st.form_submit_button(
            "🚀 Predict Employee Attrition",
            use_container_width=True,
            type="primary"
        )

    if submitted:

        try:

            probability, prediction, risk = make_prediction(
                numeric_values,
                categorical_values
            )

            record = {
                "Date & Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Prediction": prediction,
                "Risk Level": risk,
                "Leave Probability (%)": round(
                    probability * 100, 2
                ),
                "Stay Probability (%)": round(
                    (1 - probability) * 100, 2
                )
            }

            record.update(numeric_values)
            record.update(categorical_values)

            st.session_state.prediction_history.append(record)

            st.divider()

            st.markdown("### Prediction Results")

            r1, r2, r3 = st.columns(3)

            r1.metric("Prediction", prediction)
            r2.metric("Risk Level", risk)
            r3.metric(
                "Leave Probability",
                f"{probability * 100:.2f}%"
            )

            if probability >= 0.5:

                st.warning(
                    "The model estimates a higher likelihood "
                    "of employee attrition."
                )

            else:

                st.success(
                    "The model estimates a lower likelihood "
                    "of employee attrition."
                )

            st.markdown("### Probability Analysis")

            p1, p2 = st.columns(2)

            with p1:

                st.write(
                    f"**Leave: {probability * 100:.2f}%**"
                )

                st.progress(probability)

            with p2:

                stay_probability = 1 - probability

                st.write(
                    f"**Stay: {stay_probability * 100:.2f}%**"
                )

                st.progress(stay_probability)

            result_df = pd.DataFrame([record])

            st.download_button(
                "⬇️ Download Prediction Report",
                data=result_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="attripredict_report.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.caption(
                "Risk bands are display-only: Low below 30%, "
                "Moderate from 30% to below 60%, High from 60% onwards. "
                "The ANN binary prediction threshold is 50%. "
                "These are model estimates, not definitive employee assessments."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# =================================================
# HISTORY AND ANALYTICS
# =================================================

elif page == "History & Analytics":

    st.title("📈 Prediction History & Analytics")

    history = st.session_state.prediction_history

    if not history:

        st.info(
            "No prediction history available. "
            "Make your first prediction to see analytics."
        )

    else:

        history_df = pd.DataFrame(history)

        total = len(history_df)

        may_leave = int(
            (history_df["Prediction"] == "May Leave").sum()
        )

        average_probability = history_df[
            "Leave Probability (%)"
        ].mean()

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Predictions", total)
        c2.metric("May Leave", may_leave)

        c3.metric(
            "Average Leave Probability",
            f"{average_probability:.2f}%"
        )

        st.markdown("### Prediction Distribution")

        distribution = (
            history_df["Prediction"]
            .value_counts()
            .rename_axis("Prediction")
            .reset_index(name="Count")
        )

        chart_col, table_col = st.columns(2)

        with chart_col:

            st.bar_chart(
                distribution.set_index("Prediction")
            )

        with table_col:

            st.dataframe(
                history_df[
                    [
                        "Date & Time",
                        "Prediction",
                        "Risk Level",
                        "Leave Probability (%)"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            "⬇️ Download Full History",
            data=history_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="attripredict_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Clear Session History"):

            st.session_state.prediction_history = []
            st.rerun()

        st.caption(
            "History is maintained only during the current browser session. "
            "It is not stored in a permanent database."
        )


# =================================================
# ABOUT PAGE
# =================================================

elif page == "About":

    st.title("ℹ️ About AttriPredict AI")

    st.write("""
    AttriPredict AI is an employee attrition prediction application
    powered by an Artificial Neural Network.
    """)

    st.markdown("### Technology Stack")

    st.markdown("""
    - Python
    - TensorFlow and Keras
    - Artificial Neural Network
    - Pandas and NumPy
    - Scikit-learn
    - Joblib
    - Streamlit
    """)

    st.markdown("### Model Performance")

    st.metric(
        "Integrated ANN Test Accuracy",
        "80.95%"
    )

    st.caption(
        "This is the recorded test accuracy, not a guarantee "
        "of individual prediction correctness."
    )

    st.warning("""
    **Important:** This project is designed for learning and demonstration.
    Predictions may be inaccurate or biased. They must not be used as the
    sole basis for hiring, firing, promotion, or other employment decisions.
    """)
