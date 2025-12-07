import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model():
    """Load XGBoost model and scaler from HF Hub"""
    model_path = hf_hub_download(
        repo_id="RohanManiJohn/tourism-package-best-model",
        filename="best_tourism_model.pkl"
    )
    scaler_path = hf_hub_download(
        repo_id="RohanManiJohn/tourism-package-best-model",
        filename="scaler.pkl"
    )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

st.title("🧳 Wellness Tourism Package Predictor")
st.markdown("**Powered by XGBoost MLOps Pipeline**")

# Customer inputs (key features)
col1, col2 = st.columns(2)
age = col1.slider("👴 Age", 18, 70, 35)
income = col2.slider("💰 Monthly Income", 15000, 40000, 20000)
trips = col1.slider("✈️ Annual Trips", 0, 10, 2)
passport = col2.selectbox("🛂 Has Passport?", [0, 1])
city_tier = col1.selectbox("🏙️ City Tier", [1, 2, 3])

if st.button("🎯 Predict Purchase Probability", type="primary"):
    model, scaler = load_model()
    
    # Create input matching training columns
    input_data = pd.DataFrame({
        'Age': [age], 'TypeofContact': [0], 'CityTier': [city_tier],
        'DurationOfPitch': [3.0], 'Occupation': [0], 'Gender': [0],
        'NumberOfPersonVisiting': [2], 'NumberOfFollowups': [3.0],
        'ProductPitched': [0], 'PreferredPropertyStar': [3.0],
        'MaritalStatus': [0], 'NumberOfTrips': [trips], 'Passport': [passport],
        'PitchSatisfactionScore': [2], 'OwnCar': [0], 
        'NumberOfChildrenVisiting': [0.0], 'Designation': [0], 
        'MonthlyIncome': [income]
    })[X_train.columns]  # Exact column order
    
    # Predict
    X_scaled = scaler.transform(input_data)
    prob = model.predict_proba(X_scaled)[0, 1]
    
    st.success(f"**Purchase Probability: {prob:.1%}** 🎉")
    if prob > 0.5:
        st.balloons()
    else:
        st.warning("💡 Target high-income customers with passports!")
