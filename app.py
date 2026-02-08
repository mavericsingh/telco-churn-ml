import streamlit as st
import pandas as pd
import pickle
import os

from utils.preprocessing import clean_data, split_features_target
from utils.evaluation import evaluate_model, get_confusion_matrix

# --------------------------------------------------
# Streamlit Page Config (Cloud-safe)
# --------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn - Model Evaluation",
    layout="wide"
)

st.title("Telco Customer Churn – Model Evaluation Dashboard")
st.write(
    """
    This application allows evaluation of multiple trained machine learning models 
    on a user-uploaded test dataset.  
    Models were trained offline and are loaded here only for inference and evaluation.
    """
)

# --------------------------------------------------
# Model Registry
# --------------------------------------------------
MODEL_PATHS = {
    "Logistic Regression": "artifacts/logistic_regression.pkl",
    "Decision Tree": "artifacts/decision_tree.pkl",
    "KNN": "artifacts/knn.pkl",
    "Naive Bayes": "artifacts/naive_bayes.pkl",
    "Random Forest (Ensemble)": "artifacts/random_forest.pkl",
    "XGBoost (Ensemble)": "artifacts/xgboost.pkl",
}

MODEL_OPTIONS = ["-- Select Model --"] + list(MODEL_PATHS.keys())

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("Configuration")

selected_model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    MODEL_OPTIONS,
    index=0
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

submit_clicked = st.sidebar.button("Evaluate Model")


st.sidebar.markdown("### 📥 Sample Test Dataset")

with open("data/telco_test_data.csv", "rb") as f:
    st.sidebar.download_button(
        label="Download Test Dataset (CSV)",
        data=f,
        file_name="telco_test_data.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# Load Model (Lazy Loading)
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if submit_clicked:

    if selected_model_name == "-- Select Model --":
        st.error("❌ Please select a machine learning model.")
        st.stop()

    if uploaded_file is None:
        st.error("❌ Please upload a test dataset (CSV).")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)

        if "Churn" not in df.columns:
            st.error("❌ Uploaded dataset must contain the 'Churn' column.")
            st.stop()

        # st.subheader("📁 Uploaded Dataset Preview")
        # st.dataframe(df.head())

        df = clean_data(df)
        X, y = split_features_target(df)

        model_path = MODEL_PATHS[selected_model_name]
        model = load_model(model_path)

        # 🔄 Spinner added
        with st.spinner("⏳ Evaluating model on test dataset..."):
            metrics = evaluate_model(model, X, y)
            cm_df = get_confusion_matrix(model, X, y)

        st.subheader(f"📌 Evaluation Results – {selected_model_name}")

        col1, col2 = st.columns(2)

        # ---------------- Metrics Table ----------------
        with col1:
            st.markdown("### 🔢 Performance Metrics")

            metrics_df = pd.DataFrame(
                metrics.items(),
                columns=["Metric", "Value"]
            )

            center_col = st.columns([1, 3, 1])[1]
            with center_col:
                st.dataframe(
                    metrics_df,
                    hide_index=True,
                    #use_container_width=True,
                    width="stretch"
                )

            st.download_button(
                "⬇️ Download Metrics (CSV)",
                data=metrics_df.to_csv(index=False),
                file_name=f"{selected_model_name}_metrics.csv",
                mime="text/csv"
            )

        # ---------------- Confusion Matrix ----------------
        with col2:
            st.markdown("### 📉 Confusion Matrix")

            center_col = st.columns([1, 3, 1])[1]
            with center_col:
                st.table(cm_df)

            st.download_button(
                "⬇️ Download Confusion Matrix (CSV)",
                data=cm_df.to_csv(),
                file_name=f"{selected_model_name}_confusion_matrix.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error("❌ An error occurred during evaluation.")
        st.exception(e)

else:
    st.info(
        "Please select a model, upload a test dataset, and click **Evaluate Model** to view results."
    )
