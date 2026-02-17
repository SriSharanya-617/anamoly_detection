import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Anomaly Detection System", layout="wide")

st.title("🔍 Anomaly Detection System")
st.markdown("This app uses Isolation Forest to detect anomalies in air quality data.")

# ---------------- LOAD MODEL ----------------
model_path = "isolation_forest_model.pkl"

try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully")
except:
    st.error("Model file not found. Keep 'isolation_forest_model.pkl' in same folder.")
    st.stop()

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])

# =========================================================
# TAB 1 — SINGLE PREDICTION
# =========================================================
with tab1:
    st.subheader("Single Prediction")

    col1, col2 = st.columns(2)

    with col1:
        co = st.number_input("CO(GT)", value=2.0)
        nox = st.number_input("NOx(GT)", value=150.0)

    with col2:
        c6h6 = st.number_input("C6H6(GT)", value=10.0)
        no2 = st.number_input("NO2(GT)", value=100.0)

    if st.button("Predict"):
        input_df = pd.DataFrame({
            'CO(GT)': [co],
            'C6H6(GT)': [c6h6],
            'NOx(GT)': [nox],
            'NO2(GT)': [no2]
        })

        pred = model.predict(input_df)[0]
        score = model.score_samples(input_df)[0]

        if pred == -1:
            st.error("ANOMALY DETECTED")
        else:
            st.success("NORMAL")

        st.metric("Anomaly Score", round(score, 4))
        st.write(input_df)

# =========================================================
# TAB 2 — BATCH PREDICTION
# =========================================================
with tab2:
    st.subheader("Upload CSV")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        required = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

        if all(col in df.columns for col in required):

            if st.button("Predict Batch"):
                preds = model.predict(df[required])
                scores = model.score_samples(df[required])

                df['Prediction'] = preds
                df['Score'] = scores
                df['Status'] = df['Prediction'].apply(
                    lambda x: "Anomaly" if x == -1 else "Normal"
                )

                st.dataframe(df)

                normal_count = (df['Prediction'] == 1).sum()
                anomaly_count = (df['Prediction'] == -1).sum()

                col1, col2 = st.columns(2)
                col1.metric("Normal", normal_count)
                col2.metric("Anomalies", anomaly_count)

                # Plot
                fig, ax = plt.subplots()
                ax.hist(scores[preds==1], alpha=0.6, label="Normal")
                ax.hist(scores[preds==-1], alpha=0.6, label="Anomaly")
                ax.legend()
                st.pyplot(fig)

                # Download
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "results.csv")

        else:
            st.error("CSV must contain columns: CO(GT), C6H6(GT), NOx(GT), NO2(GT)")

# =========================================================
# TAB 3 — MODEL INFO
# =========================================================
with tab3:
    st.subheader("Model Info")

    st.write("Model: Isolation Forest")

    st.write("Parameters:")
    for k,v in model.get_params().items():
        st.write(f"{k}: {v}")

    st.info("""
Prediction = 1 → Normal  
Prediction = -1 → Anomaly  
Lower score → more anomalous
""")