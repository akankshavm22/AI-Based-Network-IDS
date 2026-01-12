import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
from groq import Groq
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI-NIDS | Advanced Student Project",
    layout="wide"
)

st.title("🛡️ AI-Based Network Intrusion Detection System")
st.caption("Random Forest + Explainable AI (Groq)")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("🔑 API Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

st.sidebar.header("📁 Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CIC-IDS CSV",
    type=["csv"]
)

st.sidebar.header("🤖 Model Control")
train_btn = st.sidebar.button("🚀 Train Model")

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Fwd Packet Length Max',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow Packets/s'
]

TARGET = "Label"

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred)
    }

    joblib.dump(model, "nids_model.pkl")
    return model, X_test, y_test, y_prob, metrics

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
if uploaded_file:
    df = load_data(uploaded_file)
    st.success(f"Dataset Loaded — {len(df)} packets")

    if train_btn:
        with st.spinner("Training Advanced NIDS Model..."):
            model, X_test, y_test, y_prob, metrics = train_model(df)
            st.session_state.update({
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "y_prob": y_prob,
                "metrics": metrics
            })
            st.success("Model Training Complete!")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if "model" in st.session_state:
    st.header("📊 Threat Detection Dashboard")

    col1, col2 = st.columns(2)

    # ---- RANDOM PACKET ----
    if col1.button("🎯 Capture Live Packet"):
        idx = np.random.randint(len(st.session_state["X_test"]))
        packet = st.session_state["X_test"].iloc[idx]
        true_label = st.session_state["y_test"].iloc[idx]

        st.session_state["packet"] = packet
        st.session_state["true_label"] = true_label

    # ---- DISPLAY ----
    if "packet" in st.session_state:
        packet = st.session_state["packet"]

        col1.subheader("📦 Packet Features")
        col1.dataframe(packet)

        prediction = st.session_state["model"].predict([packet])[0]
        proba = max(st.session_state["model"].predict_proba([packet])[0])

        col2.subheader("🚨 Detection Result")

        if prediction == "BENIGN":
            col2.success("BENIGN TRAFFIC")
        else:
            col2.error(f"ATTACK DETECTED — {prediction}")

        col2.metric("Attack Confidence", f"{proba*100:.2f}%")
        col2.caption(f"Actual Label: {st.session_state['true_label']}")

        # ---- FEATURE IMPORTANCE ----
        st.subheader("🔍 Feature Contribution")
        importance = st.session_state["model"].feature_importances_
        imp_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(imp_df.set_index("Feature"))

        # ---- AI EXPLANATION ----
        st.subheader("🧠 AI Security Analyst (Groq)")
        if st.button("Explain Attack"):
            if not groq_api_key:
                st.warning("Enter Groq API key")
            else:
                client = Groq(api_key=groq_api_key)

                prompt = f"""
You are a SOC analyst.

Prediction: {prediction}
Confidence: {proba:.2f}

Packet Features:
{packet.to_string()}

Explain:
• Why this traffic looks {prediction}
• Which 2-3 features matter most
• Severity level (Low/Medium/High)
• One mitigation step
Use simple student-friendly language.
"""

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5
                )

                st.info(response.choices[0].message.content)

    # ---- MODEL METRICS ----
    st.header("📈 Model Performance")

    st.metric("Accuracy", f"{st.session_state['metrics']['accuracy']*100:.2f}%")

    st.text("Classification Report")
    st.json(st.session_state["metrics"]["report"])

    st.text("Confusion Matrix")
    st.write(st.session_state["metrics"]["confusion"])

else:
    st.info("Upload dataset & train model to begin.")
