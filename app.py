import streamlit as st
import pandas as pd
from predict import predict_single_customer

st.title("🏆 Customer Tier Prediction")

st.markdown(" The goal of this project is to classify customers into marketing tiers such as **Gold 🥇, Silver 🥈, and Bronze 🥉** based on demographic and policy-related data. This classification helps in targeted marketing strategies and personalized customer engagement. 📊")

st.markdown("----")

# Collect user inputs
gender = st.selectbox("👤 Gender", ["Male", "Female"])
age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
Driving_License = st.selectbox("📄 Driving License", ["Yes", "No"])
Previously_Insured = st.selectbox("✅ Previously Insured", ["Yes", "No"])
Vehicle_Age = st.selectbox("🚗 Vehicle Age", ['> 2 Years', '1-2 Year', '< 1 Year'])
Vehicle_Damage = st.selectbox("🛠️ Vehicle Damage", ["Yes", "No"])
region_code = st.number_input("🏙️ Region Code", min_value=0, max_value=53, value=28)
premium = st.number_input("💵 Premium", min_value=0.0, value=20000.0)
vintage = st.number_input("📅 Vintage", min_value=0, max_value=300, value=100)
customer_note = st.text_area("✉️ Customer Note", value="Customer requested a callback to discuss bundling home and vehicle insurance. Strong cross-sell potential.")

# Convert Yes/No → 1/0
Driving_License = 1 if Driving_License == "Yes" else 0
Previously_Insured = 1 if Previously_Insured == "Yes" else 0

# Button to predict
if st.button("Predict Customer Tier"):

    input_df = pd.DataFrame([{
        "Gender": gender,
        "Driving_License": Driving_License,
        "Previously_Insured": Previously_Insured,
        "Vehicle_Age": Vehicle_Age,
        "Vehicle_Damage": Vehicle_Damage,
        "Region_Code": region_code,
        "Age": age,
        "Annual_Premium": premium,
        "Vintage": vintage,
        "Customer_Note": customer_note
    }])

    # Make prediction
    result = predict_single_customer(input_df)

    if result["success"]:
        st.success(f"Predicted Customer Tier: {result['tier']} ✅")
        st.write(f"🔎 Confidence: {result['confidence']:.2f}")
        st.write(f"📝 Processed Note: {result['processed_note']}")
    else:
        st.error(f"Prediction failed: {result['error']}")
