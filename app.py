import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Air Quality Health Impact Classifier")

PM25 = st.number_input("PM2.5", 0.0)
PM10 = st.number_input("PM10", 0.0)
NO2 = st.number_input("NO2", 0.0)
SO2 = st.number_input("SO2", 0.0)
CO = st.number_input("CO", 0.0)
O3 = st.number_input("O3", 0.0)

if st.button("Predict"):
    # Same order as training
    X = np.array([[PM25, PM10, NO2, SO2, CO, O3]])

    # Scale the input
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]

    st.success(f"Predicted Health Impact Class: {pred}")
