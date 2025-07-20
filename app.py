import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Healthcare Fraud Detection", layout="centered")
st.title("🩺 Healthcare Insurance Fraud Detection")
st.write("Enter patient and claim details below to predict whether it's potentially fraudulent.")

# Form Input UI
def get_user_input():
    st.header("Claim & Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        InscClaimAmtReimbursed = st.number_input("Claim Amount Reimbursed ($)", min_value=0.0, value=100.0)
        DeductibleAmtPaid = st.number_input("Deductible Amount Paid ($)", min_value=0.0, value=100.0)
        Gender = st.radio("Gender", ["Female", "Male"])
        Gender = 0 if Gender == "Female" else 1
        Race = st.selectbox("Race (Encoded)", options=[0, 1, 2, 3, 4])
        State = st.number_input("State (Encoded)", min_value=0, value=10)

    with col2:
        County = st.number_input("County ID (Encoded)", min_value=0, value=1)
        ChronicCond_Heartfailure = st.selectbox("Heart Failure", [0, 1, 2])
        ChronicCond_Cancer = st.selectbox("Cancer", [0, 1, 2])
        ChronicCond_Osteoporasis = st.selectbox("Osteoporosis", [0, 1, 2])
        ClaimDuration = st.slider("Claim Duration (days)", 0, 100, 10)
        HospitalStayDuration = st.slider("Hospital Stay Duration (days)", 0, 100, 5)

    data = {
        'InscClaimAmtReimbursed': InscClaimAmtReimbursed,
        'DeductibleAmtPaid': DeductibleAmtPaid,
        'Gender': Gender,
        'Race': Race,
        'State': State,
        'County': County,
        'ChronicCond_Heartfailure': ChronicCond_Heartfailure,
        'ChronicCond_Cancer': ChronicCond_Cancer,
        'ChronicCond_Osteoporasis': ChronicCond_Osteoporasis,
        'ClaimDuration': ClaimDuration,
        'HospitalStayDuration': HospitalStayDuration
    }

    return pd.DataFrame([data])

# Predict
input_df = get_user_input()

if st.button("🔍 Predict Fraud Status"):
    try:
        input_df = input_df[model.feature_names_in_]
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        if prediction == 1:
            st.error(f"⚠️ Likely Fraudulent Claim (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f"✅ Likely Not Fraudulent (Confidence: {prob[0]*100:.2f}%)")

        st.markdown("---")
        st.markdown("#### Prediction Probabilities")
        st.write(f"Not Fraudulent: **{prob[0]*100:.2f}%**")
        st.write(f"Fraudulent: **{prob[1]*100:.2f}%**")

    except Exception as e:
        st.error("An error occurred during prediction. Please check input data.")
