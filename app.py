import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Assignment 2 – Bank Marketing", layout="wide")

st.title("Bank Marketing Classification Demo")
st.markdown("Upload test CSV → select model → see predictions & metrics")

model_list = [
    "Logistic_Regression", "Decision_Tree", "kNN",
    "Naive_Bayes", "Random_Forest", "XGBoost"
]
uploaded = st.file_uploader("Upload CSV (same columns as bank-full.csv)", type="csv")

chosen_model = st.selectbox("Choose Model", model_list)



if uploaded is not None:
    try:
        # Try comma first, fallback to semicolon
        try:
            data = pd.read_csv(uploaded)
        except:
            uploaded.seek(0)
            data = pd.read_csv(uploaded, sep=';')

        st.subheader("Data Preview")
        st.dataframe(data.head(10))

        has_y = 'y' in data.columns

        if has_y:
            y_real = data['y'].map({'yes':1, 'no':0, 1:1, 0:0})
            X = data.drop('y', axis=1)
            st.success("Target 'y' found → metrics enabled")
        else:
            X = data
            y_real = None
            st.info("No 'y' column → predictions only")

        model_file = f"model/{chosen_model}.pkl"

        if not os.path.exists(model_file):
            st.error(f"Model missing: {model_file}")
            st.info("Check that 'model/' folder contains all .pkl files in GitHub repo")
        else:
            with st.spinner("Loading model..."):
                pipeline = joblib.load(model_file)

            preds = pipeline.predict(X)

            try:
                probs = pipeline.predict_proba(X)[:, 1]
            except:
                probs = None

            st.subheader("Predictions")
            result_df = pd.DataFrame({"Prediction": np.where(preds == 1, "yes", "no")})
            if has_y:
                result_df["Actual"] = np.where(y_real == 1, "yes", "no")
            st.dataframe(result_df.head(25))

            if has_y:
                # Remove any NaN in y_real (safety)
                valid = y_real.notna()
                y_clean = y_real[valid]
                p_clean = preds[valid]
                prob_clean = probs[valid] if probs is not None else None

                if len(y_clean) == 0:
                    st.error("No valid target values after cleaning NaN")
                else:
                    st.subheader("Metrics")

                    m = {
                        "Accuracy": accuracy_score(y_clean, p_clean),
                        "AUC": roc_auc_score(y_clean, prob_clean) if prob_clean is not None else "N/A",
                        "Precision": precision_score(y_clean, p_clean, zero_division=0),
                        "Recall": recall_score(y_clean, p_clean, zero_division=0),
                        "F1": f1_score(y_clean, p_clean, zero_division=0),
                        "MCC": matthews_corrcoef(y_clean, p_clean)
                    }

                    cols = st.columns(3)
                    for i, (k, v) in enumerate(m.items()):
                        if isinstance(v, float):
                            cols[i % 3].metric(k, f"{v:.4f}")
                        else:
                            cols[i % 3].metric(k, v)

                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_clean, p_clean)
                    fig, ax = plt.subplots(figsize=(5,4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['no','yes'], yticklabels=['no','yes'])
                    st.pyplot(fig)

                    st.subheader("Classification Report")
                    st.text(classification_report(y_clean, p_clean, target_names=['no','yes']))

    except Exception as e:
        st.error("Error during processing")

        st.exception(e)
