import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("Machine Learning Classification Model Comparison")
st.write("Upload test dataset and select model for prediction.")

# Load scaler
scaler = joblib.load(open("model/scaler.joblib", "rb"))

# Model dictionary
model = {
    "Logistic Regression": "Logistic_Regression.joblib",
    "Decision Tree": "Decision_Tree.joblib",
    "KNN": "kNN.joblib",
    "Naive Bayes": "Naive_Bayes.joblib",
    "Random Forest": "Random_Forest.joblib",
    "XGBoost": "XGBoost.joblib"
}

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

model_name = st.selectbox("Select Model", list(model_files.keys()))

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(data.head())

    if st.button("Run Model"):

        model = pickle.load(open(f"model/{model_files[model_name]}", "rb"))

        # Assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Scale only for LR and KNN
        if model_name in ["Logistic Regression", "KNN"]:
            X = scaler.transform(X)

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_prob)
        else:
            auc = roc_auc_score(y, y_pred)

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
        col2.metric("Precision", round(precision_score(y, y_pred), 4))
        col3.metric("Recall", round(recall_score(y, y_pred), 4))

        col1.metric("F1 Score", round(f1_score(y, y_pred), 4))
        col2.metric("MCC", round(matthews_corrcoef(y, y_pred), 4))
        col3.metric("AUC", round(auc, 4))

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))
