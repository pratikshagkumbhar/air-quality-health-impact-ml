import streamlit as st
import pickle
import numpy as np

# Load saved model
model = pickle.load(open("model.pkl", "rb"))

st.title("Air Quality Health Impact Prediction")
st.write("This app predicts HealthImpactClass (0–4) based on air quality measurements.")

# Input fields
pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
so2 = st.number_input("SO2")
co = st.number_input("CO")

if st.button("Predict"):
    X = np.array([[pm25, pm10, no2, so2, co]])
    pred = model.predict(X)[0]
    st.success(f"Predicted Health Impact Class: {pred}")
