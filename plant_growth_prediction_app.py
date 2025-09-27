# %%
import streamlit as st
import pandas as pd
import pickle

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
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Enter Plant Details")

# Numerical inputs
sunlight = st.sidebar.slider("Sunlight Hours (per day)", min_value=1.0, max_value=12.0, value=6.0, step=0.1)
temperature = st.sidebar.slider("Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0, step=0.1)
humidity = st.sidebar.slider("Humidity (%)", min_value=20.0, max_value=100.0, value=60.0, step=0.1)

# Categorical inputs
soil_type = st.sidebar.selectbox("Soil Type", options=["loam", "sandy", "clay"])
water_frequency = st.sidebar.selectbox("Water Frequency", options=["daily", "weekly", "bi-weekly"])
fertilizer_type = st.sidebar.selectbox("Fertilizer Type", options=["organic", "chemical", "none"])

# -------------------------------
# Preprocessing (one-hot encoding to match training)
# -------------------------------
def preprocess_input(sunlight, temperature, humidity, soil_type, water_frequency, fertilizer_type):
    data = {
        "Sunlight_Hours": sunlight,
        "Temperature": temperature,
        "Humidity": humidity,
        # One-hot encoded categories (must match training names!)
        "Soil_Type_clay": 1 if soil_type == "clay" else 0,
        "Soil_Type_loam": 1 if soil_type == "loam" else 0,
        "Soil_Type_sandy": 1 if soil_type == "sandy" else 0,
        "Water_Frequency_daily": 1 if water_frequency == "daily" else 0,
        "Water_Frequency_weekly": 1 if water_frequency == "weekly" else 0,
        "Water_Frequency_bi-weekly": 1 if water_frequency == "bi-weekly" else 0,
        "Fertilizer_Type_organic": 1 if fertilizer_type == "organic" else 0,
        "Fertilizer_Type_chemical": 1 if fertilizer_type == "chemical" else 0,
        "Fertilizer_Type_none": 1 if fertilizer_type == "none" else 0,
    }
    return pd.DataFrame([data])

# -------------------------------
# Prediction Button
# -------------------------------
if st.sidebar.button("Predict"):
    input_df = preprocess_input(sunlight, temperature, humidity, soil_type, water_frequency, fertilizer_type)

    try:
        prediction = logreg.predict(input_df)[0]
        st.subheader("🌿 Prediction Result")
        if prediction == 1:
            st.success("The plant is predicted to **reach growth milestone** ✅")
        else:
            st.warning("The plant is predicted to **not reach growth milestone** ❌")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# -------------------------------
# Instructions
# -------------------------------
st.write("""
### 📌 Instructions
1. Enter plant details in the sidebar.
2. Click **Predict** to get the growth milestone prediction.
3. Adjust inputs to compare different conditions.
""")


