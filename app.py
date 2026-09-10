
import streamlit as st
import pandas as pd
import joblib
import json

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Vendor Risk Scorecard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Load Model Files
# --------------------------------------------------

model = joblib.load("vendor_risk_model.pkl")
scaler = joblib.load("vendor_risk_scaler.pkl")

with open("model_features.json", "r") as file:
    model_features = json.load(file)

vendor_ranking = pd.read_csv("vendor_ranking.csv")

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📊 Vendor Risk Scorecard")
st.write(
    "Machine Learning based vendor risk assessment "
    "for procurement decision support."
)

st.divider()

# --------------------------------------------------
# Vendor Risk Prediction
# --------------------------------------------------

st.header("🔍 Predict Vendor Risk")

col1, col2 = st.columns(2)

with col1:
    defect_rate = st.number_input(
        "Defect Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.1
    )

    delivery_days = st.number_input(
        "Delivery Days",
        min_value=0.0,
        max_value=365.0,
        value=11.0,
        step=1.0
    )

with col2:
    price_savings = st.number_input(
        "Price Savings (%)",
        min_value=0.0,
        max_value=100.0,
        value=8.0,
        step=0.1
    )

    compliance = st.selectbox(
        "Compliance",
        ["Yes", "No"]
    )

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("Predict Vendor Risk", type="primary"):

    # Convert percentage to decimal
    defect_rate_decimal = defect_rate / 100

    compliance_value = 1 if compliance == "Yes" else 0

    # Create input in the exact feature order
    input_data = pd.DataFrame({
        "Defect_Rate": [defect_rate_decimal],
        "Delivery_Days": [delivery_days],
        "Price_Savings_Percent": [price_savings],
        "Compliance": [compliance_value]
    })

    # Scale input using saved scaler
    input_scaled = scaler.transform(input_data)

    # Predict class
    prediction = model.predict(input_scaled)[0]

    # Prediction probability
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🔴 HIGH RISK")
        st.write(
            f"High-risk probability: **{probability * 100:.2f}%**"
        )
        st.warning(
            "Recommended action: closely monitor this vendor "
            "and review quality and compliance performance."
        )
    else:
        st.success("🟢 LOW RISK")
        st.write(
            f"High-risk probability: **{probability * 100:.2f}%**"
        )
        st.info(
            "Recommended action: vendor performance appears "
            "relatively stable based on the provided profile."
        )

# --------------------------------------------------
# Vendor Ranking
# --------------------------------------------------

st.divider()

st.header("🏆 Vendor Risk Ranking")

st.dataframe(
    vendor_ranking,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# Procurement Recommendations
# --------------------------------------------------

st.divider()

st.header("📋 Procurement Recommendations")

recommendations = {
    "Delta_Logistics":
        "High priority monitoring; improve quality control and compliance before increasing orders.",

    "Beta_Supplies":
        "Close monitoring; focus on reducing defects and improving compliance.",

    "Gamma_Co":
        "Moderate monitoring; improve defect control and maintain compliance.",

    "Epsilon_Group":
        "Preferred supplier; maintain current performance and compliance.",

    "Alpha_Inc":
        "Preferred supplier; strong quality performance with low overall risk."
}

for supplier, recommendation in recommendations.items():

    st.write(f"**{supplier}**")

    if "High priority" in recommendation or "Close monitoring" in recommendation:
        st.warning(recommendation)
    elif "Moderate" in recommendation:
        st.info(recommendation)
    else:
        st.success(recommendation)

# --------------------------------------------------
# Project Information
# --------------------------------------------------

st.divider()

st.caption(
    "Vendor Risk Scorecard | Logistic Regression | "
    "Explainable AI using SHAP"
)
