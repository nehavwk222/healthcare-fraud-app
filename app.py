import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("random_forest_model.pkl")

# Set up encoders with same values used in training
state_encoder = LabelEncoder()
race_encoder = LabelEncoder()

state_encoder.fit([
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
])
race_encoder.fit(["White", "Black", "Asian", "Hispanic", "Other"])

st.set_page_config(page_title="Healthcare Fraud Detection", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    body { background-color: #f8fbff; }
    .main { background-color: #ffffff; color: #003366; }
    .st-bw { color: #003366; }
    .st-c3 { color: #005792; }
    .st-af { background-color: #cce6ff !important; border-radius: 8px; padding: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Healthcare Insurance Fraud Detection")
st.subheader("Enter patient and claim details to check for potential fraud")

# Form Input UI
def get_user_input():
    with st.form(key="input_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💳 Financial Details")
            IPAnnualReimbursementAmt = st.number_input("Annual Reimbursement Amount ($)", min_value=0.0, value=1000.0)
            IPAnnualDeductibleAmt = st.number_input("Annual Deductible Amount ($)", min_value=0.0, value=500.0)
            InscClaimAmtReimbursed = st.number_input("Claim Amount Reimbursed ($)", min_value=0.0, value=100.0)
            DeductibleAmtPaid = st.number_input("Deductible Amount Paid ($)", min_value=0.0, value=100.0)
            ClaimDuration = st.slider("Claim Duration (days)", 0, 100, 10)
            HospitalStayDuration = st.slider("Hospital Stay Duration (days)", 0, 100, 5)

        with col2:
            st.markdown("### 🧑‍⚕️ Patient Info")
            Gender = st.radio("Gender", ["Female", "Male"])
            Gender = 0 if Gender == "Female" else 1
            NoOfMonths_PartACov = st.slider("Months of Part A Coverage", 0, 12, 12)
            NoOfMonths_PartBCov = st.slider("Months of Part B Coverage", 0, 12, 12)
            State = st.selectbox("State", state_encoder.classes_.tolist())
            County = st.text_input("County (for numeric ID)", value="Default County")
            Race = st.selectbox("Race", race_encoder.classes_.tolist())

        st.markdown("### 🩹 Chronic Conditions")
        yn_map = {"No": 0, "Yes": 1}
        chronic_features = {
            "ChronicCond_Alzheimer": st.selectbox("Alzheimer", ["No", "Yes"]),
            "ChronicCond_Heartfailure": st.selectbox("Heart Failure", ["No", "Yes"]),
            "ChronicCond_KidneyDisease": st.selectbox("Kidney Disease", ["No", "Yes"]),
            "ChronicCond_Cancer": st.selectbox("Cancer", ["No", "Yes"]),
            "ChronicCond_ObstrPulmonary": st.selectbox("Pulmonary Disease", ["No", "Yes"]),
            "ChronicCond_Depression": st.selectbox("Depression", ["No", "Yes"]),
            "ChronicCond_Diabetes": st.selectbox("Diabetes", ["No", "Yes"]),
            "ChronicCond_IschemicHeart": st.selectbox("Ischemic Heart", ["No", "Yes"]),
            "ChronicCond_Osteoporasis": st.selectbox("Osteoporosis", ["No", "Yes"]),
            "ChronicCond_rheumatoidarthritis": st.selectbox("Rheumatoid Arthritis", ["No", "Yes"]),
            "ChronicCond_stroke": st.selectbox("Stroke", ["No", "Yes"]),
        }

        submit_button = st.form_submit_button(label="🔍 Predict Fraud Status")

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

    return pd.DataFrame([data]), submit_button

# Run the form and prediction
input_df, submitted = get_user_input()

if submitted:
    input_df = input_df[model.feature_names_in_]
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ Likely Fraudulent Claim (Confidence: {prob[1]*100:.2f}%)")
    else:
        st.success(f"✅ Likely Not Fraudulent (Confidence: {prob[0]*100:.2f}%)")

    st.markdown("### 📊 Prediction Probabilities")
    st.info(f"Not Fraudulent: **{prob[0]*100:.2f}%**")
    st.info(f"Fraudulent: **{prob[1]*100:.2f}%**")
