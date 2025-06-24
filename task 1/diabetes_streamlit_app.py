import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter your health information below to predict if you're diabetic.")

# Define input fields
features = {
    "Pregnancies": st.number_input("Pregnancies", min_value=0, max_value=20, value=2),
    "Glucose": st.number_input("Glucose", min_value=0, max_value=300, value=120),
    "BloodPressure": st.number_input("Blood Pressure", min_value=0, max_value=200, value=70),
    "SkinThickness": st.number_input("Skin Thickness", min_value=0, max_value=100, value=25),
    "Insulin": st.number_input("Insulin", min_value=0, max_value=900, value=80),
    "BMI": st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0),
    "DiabetesPedigreeFunction": st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5),
    "Age": st.number_input("Age", min_value=0, max_value=120, value=30)
}

# Prepare input for prediction
input_data = np.array([list(features.values())]).reshape(1, -1)
input_scaled = scaler.transform(input_data)

# Predict on button click
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("🔴 The model predicts that the person is **diabetic**.")
    else:
        st.success("🟢 The model predicts that the person is **not diabetic**.")
