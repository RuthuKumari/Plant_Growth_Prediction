# %%

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Plant Growth Prediction", page_icon="🌱")
st.title("🌱 Plant Growth Prediction App")

# -------------------------------
# Load trained logistic regression model
# -------------------------------
try:
    with open("plant_logreg_model.pkl", "rb") as file:
        logreg = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'plant_logreg_model.pkl' not found. Please upload it to the same directory.")
    st.stop()

# -------------------------------
# Sidebar Inputs (Customize features here)
# -------------------------------
st.sidebar.header("Enter Plant Details")

# Example features — change these to match your dataset
sunlight = st.sidebar.slider("Sunlight (hours/day)", min_value=1, max_value=12, value=6)
water = st.sidebar.slider("Water (liters/week)", min_value=1, max_value=20, value=10)
soil_ph = st.sidebar.slider("Soil pH", min_value=3.0, max_value=9.0, value=6.5, step=0.1)
fertilizer = st.sidebar.slider("Fertilizer Amount (kg/acre)", min_value=0, max_value=100, value=30)
temperature = st.sidebar.slider("Temperature (°C)", min_value=10, max_value=40, value=25)

soil_type = st.sidebar.selectbox("Soil Type", options=["Sandy", "Clay", "Loamy"])
fertilizer_type = st.sidebar.selectbox("Fertilizer Type", options=["Organic", "Chemical", "Mixed"])

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_input(sunlight, water, soil_ph, fertilizer, temperature, soil_type, fertilizer_type):
    data = {
        "Sunlight": sunlight,
        "Water": water,
        "Soil_pH": soil_ph,
        "Fertilizer": fertilizer,
        "Temperature": temperature,
        "SoilType_Sandy": 1 if soil_type == "Sandy" else 0,
        "SoilType_Clay": 1 if soil_type == "Clay" else 0,
        "SoilType_Loamy": 1 if soil_type == "Loamy" else 0,
        "FertilizerType_Organic": 1 if fertilizer_type == "Organic" else 0,
        "FertilizerType_Chemical": 1 if fertilizer_type == "Chemical" else 0,
        "FertilizerType_Mixed": 1 if fertilizer_type == "Mixed" else 0,
    }
    return pd.DataFrame([data])

# -------------------------------
# Prediction Button
# -------------------------------
if st.sidebar.button("Predict"):
    input_df = preprocess_input(sunlight, water, soil_ph, fertilizer, temperature, soil_type, fertilizer_type)

    try:
        prediction = logreg.predict(input_df)[0]

        st.subheader("🌿 Prediction Result")
        st.write(f"Predicted Plant Growth Category: **{prediction}**")

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# -------------------------------
# Instructions
# -------------------------------
st.write("""
### 📌 Instructions
1. Enter plant details in the sidebar.
2. Click **Predict** to get the growth category.
3. Adjust inputs to compare different scenarios.
""")


