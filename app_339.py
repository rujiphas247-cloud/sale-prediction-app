# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression

# Streamlit app title
st.title("Sales Prediction App")

# Step 1: Load the model
model_file = 'model-reg-xxx.pkl'
try:
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file '{model_file}' not found in the current directory.")
    model = joblib.load(model_file)
    if not isinstance(model, LinearRegression):
        raise TypeError("Loaded model is not a LinearRegression model.")
    st.success("Model loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Step 2: Create input interface for DataFrame
st.header("Enter Advertising Budgets")
expected_columns = ['youtube', 'tiktok', 'instagram']

# Create sliders for input values (default to 50, range 0-100)
youtube = st.slider("YouTube Budget", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
tiktok = st.slider("TikTok Budget", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
instagram = st.slider("Instagram Budget", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

# Create DataFrame from inputs
new_data = pd.DataFrame({
    'youtube': [youtube],
    'tiktok': [tiktok],
    'instagram': [instagram]
})

# Validate DataFrame
try:
    if not all(col in new_data.columns for col in expected_columns):
        raise ValueError(f"DataFrame must contain columns: {expected_columns}")
    if new_data.shape[1] != len(expected_columns):
        raise ValueError(f"DataFrame must have exactly {len(expected_columns)} columns: {expected_columns}")
    if not new_data.select_dtypes(include=['float64', 'int64']).columns.tolist() == expected_columns:
        raise ValueError("All columns must be numeric (float64 or int64).")
    st.write("Input data validated successfully.")
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()

# Step 3: Make predictions when button is clicked
if st.button("Predict Sales"):
    try:
        predictions = model.predict(new_data)
        st.success(f"Estimated Sales: {predictions[0]:.2f}")
    except Exception as e:
        st.error(f"Error making predictions: {e}")
