import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Encoders setup (same as training phase)
state_encoder = LabelEncoder()
race_encoder = LabelEncoder()

state_encoder.fit([
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
])
race_encoder.fit(["White", "Black", "Asian", "Hispanic", "Other"])

# Set Streamlit page config
st.set_page_config(page_title="Healthcare Fraud Detection System", layout="wide", page_icon="🩺")

# Optional banner image
st.image("https://cdn.pixabay.com/photo/2021/04/21/06/18/medical-6195053_960_720.jpg", use_column_width=True)

st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .stButton>button {
            background-color: #0b74de;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5em 2em;
        }
        .stNumberInput>div>input {
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ AI-Powered Healthcare Insurance Fraud Detection")
st.markdown("""
This AI system helps predict whether a claim is **potentially fraudulent** or **legitimate** based on claim metadata.
Use the form below to simulate a new claim prediction.
---
""")

# Input form section
def get_user_input():
    with st.form("claim_form"):
        st.subheader("📋 Claim Details")

        col1, col2 = st.columns(2)
        with col1:
            IPAnnualReimbursementAmt = st.number_input("💵 Annual Reimbursement Amount", min_value=0.0, value=1000.0)
            IPAnnualDeductibleAmt = st.number_input("💸 Annual Deductible Amount", min_value=0.0, value=500.0)
            InscClaimAmtReimbursed = st.number_input("💰 Claim Amount Reimbursed", min_value=0.0, value=100.0)
            DeductibleAmtPaid = st.number_input("💳 Deductible Amount Paid", min_value=0.0, value=100.0)
        with col2:
            NoOfMonths_PartACov = st.slider("🕐 Months of Part A Coverage", 0, 12, 12)
            NoOfMonths_PartBCov = st.slider("🕑 Months of Part B Coverage", 0, 12, 12)
            ClaimDuration = st.slider("📆 Claim Duration (days)", 0, 100, 10)
            HospitalStayDuration = st.slider("🏥 Hospital Stay Duration (days)", 0, 100, 5)

        st.subheader("📍 Demographics")
        col3, col4 = st.columns(2)
        with col3:
            State = st.selectbox("State", state_encoder.classes_.tolist())
            County = st.text_input("County", value="Default County")
        with col4:
            Race = st.selectbox("Race", race_encoder.classes_.tolist())
            Gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            Gender = 0 if Gender == "Female" else 1

        st.subheader("🩺 Chronic Conditions")
        chronic_features = {}
        chronic_conditions = [
            "Alzheimer", "Heartfailure", "KidneyDisease", "Cancer", "ObstrPulmonary",
            "Depression", "Diabetes", "IschemicHeart", "Osteoporasis", "rheumatoidarthritis", "stroke"
        ]
        yn_map = {"No": 0, "Yes": 1}

        for cond in chronic_conditions:
            chronic_features[f"ChronicCond_{cond}"] = st.selectbox(cond.replace("rheumatoidarthritis", "Rheumatoid Arthritis").replace("ObstrPulmonary", "Pulmonary Disease"), ["No", "Yes"], key=cond)

        submit = st.form_submit_button("🔍 Predict Fraud Status")

    data = {
        'IPAnnualReimbursementAmt': IPAnnualReimbursementAmt,
        'IPAnnualDeductibleAmt': IPAnnualDeductibleAmt,
        'InscClaimAmtReimbursed': InscClaimAmtReimbursed,
        'DeductibleAmtPaid': DeductibleAmtPaid,
        'NoOfMonths_PartACov': NoOfMonths_PartACov,
        'NoOfMonths_PartBCov': NoOfMonths_PartBCov,
        'State': state_encoder.transform([State])[0],
        'County': hash(County) % 100,
        'Race': race_encoder.transform([Race])[0],
        'Gender': Gender,
        'ClaimDuration': ClaimDuration,
        'HospitalStayDuration': HospitalStayDuration,
    }

    for k, v in chronic_features.items():
        data[k] = yn_map[v]

    return pd.DataFrame([data]), submit

# Main
input_df, submitted = get_user_input()

if submitted:
    input_df = input_df[model.feature_names_in_]
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.markdown("---")
    if prediction == 1:
        st.error(f"\n🚨 **Result: Potentially Fraudulent Claim!**\n\nConfidence: {prob[1]*100:.2f}%")
    else:
        st.success(f"\n✅ **Result: Likely Legitimate Claim**\n\nConfidence: {prob[0]*100:.2f}%")

    st.markdown("""
    <br>
    <h4>📊 Detailed Prediction Probabilities</h4>
    <ul>
        <li><b>Not Fraudulent:</b> {:.2f}%</li>
        <li><b>Fraudulent:</b> {:.2f}%</li>
    </ul>
    """.format(prob[0]*100, prob[1]*100), unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with 💙 using Streamlit and Machine Learning")
