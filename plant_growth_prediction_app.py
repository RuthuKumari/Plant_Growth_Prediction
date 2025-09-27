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
# Load training columns
# -------------------------------
try:
    with open("training_columns.pkl", "rb") as f:
        training_columns = pickle.load(f)
except FileNotFoundError:
    st.error("File 'training_columns.pkl' not found. Please save the training columns during training.")
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
# Preprocessing Function
# -------------------------------
def preprocess_input(sunlight, temperature, humidity, soil_type, water_frequency, fertilizer_type):
    # Create initial input DataFrame
    input_dict = {
        "Sunlight_Hours": [sunlight],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Soil_Type": [soil_type],
        "Water_Frequency": [water_frequency],
        "Fertilizer_Type": [fertilizer_type]
    }
    df = pd.DataFrame(input_dict)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["Soil_Type", "Water_Frequency", "Fertilizer_Type"])

    # Add missing columns from training with 0
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[training_columns]

    return df

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
