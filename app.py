import streamlit as st
import pandas as pd
import joblib

# Load models
xgb_model = joblib.load("xgboost_model.pkl")
lr_model = joblib.load("linear_regression.pkl")

# Feature columns expected by the model
features = [
    "occupancy_rate_lag1", "ami_lag1", "asset_value_lag1", "inventory_units_lag1",
    "under_construction_units_lag1", "market_asking_rent/unit_lag1",
    "market_effective_rent_growth_lag1", "market_sale_price_per_unit_lag1",
    "unemployment_rate_lag1", "market_cap_rate_lag1"
]

# Page title
st.title("🏘️ Occupancy Rate Prediction Tool")
model_choice = st.selectbox("Select Model", ["XGBoost", "Linear Regression"])

st.markdown("### 📥 Choose Input Method")

input_option = st.radio("Select how you want to input data:", ["🔢 Manual Entry", "📄 Upload CSV File"])

# --- Option 1: Manual Input
if input_option == "🔢 Manual Entry":
    inputs = {f: st.number_input(f, value=0.0) for f in features}
    input_df = pd.DataFrame([inputs])

    if st.button("🔮 Predict"):
        model = xgb_model if model_choice == "XGBoost" else lr_model
        result = model.predict(input_df)[0]
        st.success(f"📈 Predicted Occupancy Rate: **{result:.4f}**")

# --- Option 2: Upload CSV for batch prediction
else:
    uploaded_file = st.file_uploader("Upload a CSV file with all required features", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Check if all required features are present
        missing_cols = [f for f in features if f not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
        else:
            model = xgb_model if model_choice == "XGBoost" else lr_model
            df["predicted_occupancy_rate"] = model.predict(df[features])
            st.success("✅ Prediction complete!")
            st.dataframe(df)

            # Download results
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Download Results CSV", data=csv, file_name="predicted_results.csv", mime="text/csv")
