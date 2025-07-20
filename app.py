import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Encoders
gender_map = {"Female": 0, "Male": 1}
race_encoder = LabelEncoder()
state_encoder = LabelEncoder()

race_encoder.fit(["White", "Black", "Asian", "Hispanic", "Other"])
state_encoder.fit([
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
])

st.set_page_config(page_title="Healthcare Insurance Fraud Detection", layout="wide")
st.title("🩺 Healthcare Insurance Fraud Detection")
st.markdown("Enter claim and patient details to check for potential fraud.")

# Input UI
with st.form("fraud_form"):
    st.subheader("📝 Claim & Patient Info")

    col1, col2 = st.columns(2)

    with col1:
        reimbursed = st.number_input("Claim Amount Reimbursed ($)", min_value=0.0)
        deductible_paid = st.number_input("Deductible Amount Paid ($)", min_value=0.0)
        gender = st.selectbox("Gender", ["Female", "Male"])
        race = st.selectbox("Race", race_encoder.classes_)
        state = st.selectbox("State", state_encoder.classes_)
        county = st.text_input("County (any name)", value="Default County")

    with col2:
        heart_failure = st.radio("Heart Failure", ["No", "Yes"])
        cancer = st.radio("Cancer", ["No", "Yes"])
        osteoporosis = st.radio("Osteoporosis", ["No", "Yes"])
        claim_days = st.slider("Claim Duration (days)", 0, 100, 10)
        hospital_days = st.slider("Hospital Stay Duration (days)", 0, 100, 5)

    submitted = st.form_submit_button("🔍 Predict Fraud")

    if submitted:
        # Convert to model input
        input_data = {
            "InscClaimAmtReimbursed": reimbursed,
            "DeductibleAmtPaid": deductible_paid,
            "Gender": gender_map[gender],
            "Race": race_encoder.transform([race])[0],
            "State": state_encoder.transform([state])[0],
            "County": hash(county) % 100,
            "ChronicCond_Heartfailure": 1 if heart_failure == "Yes" else 0,
            "ChronicCond_Cancer": 1 if cancer == "Yes" else 0,
            "ChronicCond_Osteoporasis": 1 if osteoporosis == "Yes" else 0,
            "ClaimDuration": claim_days,
            "HospitalStayDuration": hospital_days,
        }

        df = pd.DataFrame([input_data])

        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0]

        if prediction == 1:
            st.error(f"⚠️ Potential Fraud Detected (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f"✅ No Fraud Detected (Confidence: {prob[0]*100:.2f}%)")

        st.markdown("### 🔍 Prediction Details")
        st.write(f"Not Fraudulent: **{prob[0]*100:.2f}%**")
        st.write(f"Fraudulent: **{prob[1]*100:.2f}%**")
