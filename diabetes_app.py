import streamlit as st
import joblib

# Load saved model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction Web App")
st.write("Enter the patient details below:")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose Level", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 25.5)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 0, 120, 33)

# Prediction
if st.button("Predict"):
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"The patient is **{result}**")
