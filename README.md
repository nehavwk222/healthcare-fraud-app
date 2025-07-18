# 🩺 Healthcare Insurance Fraud Detection

A Streamlit web app to predict if a healthcare insurance claim is fraudulent or not using a trained Random Forest model.

## 🚀 Features

- Input patient claim details
- Model predicts "Fraud" or "Not Fraud"
- Displays fraud probability

## 🧠 Model

- Trained on preprocessed healthcare insurance dataset
- Random Forest Classifier
- Includes feature engineering: durations, chronic conditions, etc.

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Joblib

## ▶️ Try It Live

👉 Hosted on [Streamlit Cloud](https://share.streamlit.io/...)

## 📝 How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-fraud-app.git
cd healthcare-fraud-app
pip install -r requirements.txt
streamlit run app.py
