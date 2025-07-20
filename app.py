import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Healthcare Fraud Detection", layout="centered")
st.title("🩺 Healthcare Insurance Fraud Detection")
st.write("Enter details below to predict whether the insurance claim is potentially fraudulent.")

# Form Input UI
def get_user_input():
    st.subheader("📋 Claim & Patient Information")

    InscClaimAmtReimbursed = st.number_input("Claim Amount Reimbursed ($)", min_value=0.0, value=1000.0)
    DeductibleAmtPaid = st.number_input("Deductible Amount Paid ($)", min_value=0.0, value=500.0)
    Gender = st.selectbox("Gender", ["Female", "Male"])
    Gender = 0 if Gender == "Female" else 1

    Race = st.selectbox("Race", ["White", "Black", "Asian", "Hispanic", "Other"])
    race_map = {"White": 0, "Black": 1, "Asian": 2, "Hispanic": 3, "Other": 4}
    Race = race_map[Race]

    State = st.selectbox("State", [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
        "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
        "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    ])
    State = hash(State) % 50

    County = st.text_input("County Name", value="Default County")
    County = hash(County) % 100

    st.subheader("🩺 Chronic Conditions")
    yn_map = {"No": 0, "Yes": 1, "Unknown": 2}
    ChronicCond_Heartfailure = st.selectbox("Heart Failure", ["No", "Yes", "Unknown"])
    ChronicCond_Cancer = st.selectbox("Cancer", ["No", "Yes", "Unknown"])
    ChronicCond_Osteoporasis = st.selectbox("Osteoporosis", ["No", "Yes", "Unknown"])

    ClaimDuration = st.slider("Claim Duration (days)", 0, 100, 10)
    HospitalStayDuration = st.slider("Hospital Stay Duration (days)", 0, 100, 5)

    data = {
        'InscClaimAmtReimbursed': InscClaimAmtReimbursed,
        'DeductibleAmtPaid': DeductibleAmtPaid,
        'Gender': Gender,
        'Race': Race,
        'State': State,
        'County': County,
        'ChronicCond_Heartfailure': yn_map[ChronicCond_Heartfailure],
        'ChronicCond_Cancer': yn_map[ChronicCond_Cancer],
        'ChronicCond_Osteoporasis': yn_map[ChronicCond_Osteoporasis],
        'ClaimDuration': ClaimDuration,
        'HospitalStayDuration': HospitalStayDuration
    }

    return pd.DataFrame([data])

# Predict
input_df = get_user_input()

if st.button("🔍 Predict Fraud Status"):
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
