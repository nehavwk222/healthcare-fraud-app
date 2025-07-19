import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("random_forest_model.pkl")

# Set up encoders
state_encoder = LabelEncoder()
race_encoder = LabelEncoder()

state_encoder.fit([
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
])
race_encoder.fit(["White", "Black", "Asian", "Hispanic", "Other"])

# App title and layout
st.set_page_config(page_title="Healthcare Fraud Detection", layout="centered", page_icon="🩺")

# Sidebar
with st.sidebar:
    st.image("https://i.imgur.com/OHR9gGf.png", width=200)  # Optional logo
    st.markdown("### About This App")
    st.info("Predict potential healthcare insurance fraud using a trained ML model.")

# Title
st.title("🩺 Healthcare Insurance Fraud Detection")
st.write("Enter claim details to predict if it's **potentially fraudulent.**")

# Form Input UI
def get_user_input():
    col1, col2 = st.columns(2)

    with col1:
        IPAnnualReimbursementAmt = st.number_input("Annual Reimbursement Amount", min_value=0.0, value=1000.0)
        InscClaimAmtReimbursed = st.number_input("Claim Amount Reimbursed", min_value=0.0, value=100.0)
        NoOfMonths_PartACov = st.slider("Months of Part A Coverage", 0, 12, 12)
        State = st.selectbox("State", state_encoder.classes_.tolist())
        Gender = st.selectbox("Gender", ["Female", "Male"])
        Gender = 0 if Gender == "Female" else 1

    with col2:
        IPAnnualDeductibleAmt = st.number_input("Annual Deductible Amount", min_value=0.0, value=500.0)
        DeductibleAmtPaid = st.number_input("Deductible Amount Paid", min_value=0.0, value=100.0)
        NoOfMonths_PartBCov = st.slider("Months of Part B Coverage", 0, 12, 12)
        Race = st.selectbox("Race", race_encoder.classes_.tolist())
        County = st.text_input("County (used for numeric ID)", value="Default County")

    ClaimDuration = st.slider("Claim Duration (days)", 0, 100, 10)
    HospitalStayDuration = st.slider("Hospital Stay Duration (days)", 0, 100, 5)

    st.subheader("⚕️ Chronic Conditions")
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

    return pd.DataFrame([data])

# Get input
input_df = get_user_input()

# Display input summary
st.markdown("### 📋 Input Summary")
st.dataframe(input_df)

# Prediction
if st.button("🔍 Predict Fraud Status"):
    input_df = input_df[model.feature_names_in_]
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ Likely Fraudulent Claim (Confidence: {prob[1]*100:.2f}%)")
    else:
        st.success(f"✅ Likely Not Fraudulent (Confidence: {prob[0]*100:.2f}%)")

    st.markdown("#### 📊 Prediction Probabilities")
    st.write(f"Not Fraudulent: **{prob[0]*100:.2f}%**")
    st.write(f"Fraudulent: **{prob[1]*100:.2f}%**")

    # Chart
    fig, ax = plt.subplots()
    labels = ['Not Fraudulent', 'Fraudulent']
    ax.barh(labels, prob, color=['green', 'red'])
    ax.set_xlim([0, 1])
    for i, v in enumerate(prob):
        ax.text(v + 0.01, i, f"{v*100:.2f}%", color='black', va='center')
    st.pyplot(fig)

    # Download CSV
    output = input_df.copy()
    output["Prediction"] = ["Not Fraudulent" if prediction == 0 else "Fraudulent"]
    st.download_button("📥 Download Prediction CSV", data=output.to_csv(index=False), file_name="prediction_result.csv", mime="text/csv")
