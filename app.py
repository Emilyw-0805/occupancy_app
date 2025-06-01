import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score

# Load models
xgb_model = joblib.load("xgboost_model.pkl")
lr_model = joblib.load("linear_regression.pkl")

# Expected feature columns
features = [
    "occupancy_rate_lag1", "ami_lag1", "asset_value_lag1", "inventory_units_lag1",
    "under_construction_units_lag1", "market_asking_rent/unit_lag1",
    "market_effective_rent_growth_lag1", "market_sale_price_per_unit_lag1",
    "unemployment_rate_lag1", "market_cap_rate_lag1"
]

# Page layout
st.title("🏘️ Occupancy Rate Prediction Tool")
model_choice = st.selectbox("Select Model", ["XGBoost", "Linear Regression"])

st.markdown("### 📄 Upload CSV for Batch Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with all required features", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check for missing features
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
    else:
        # Predict
        model = xgb_model if model_choice == "XGBoost" else lr_model
        df["predicted_occupancy_rate"] = model.predict(df[features])
        
        st.success("✅ Prediction complete!")
        st.dataframe(df)

        # Show R² if actuals are included
        if "occupancy_rate" in df.columns:
            r2 = r2_score(df["occupancy_rate"], df["predicted_occupancy_rate"])
            st.markdown(f"📈 **R² Score**: `{r2:.4f}`")

        # Show coefficients if linear regression
        if model_choice == "Linear Regression":
            st.markdown("### 📊 Coefficients (Linear Regression)")
            coefs = pd.Series(model.coef_, index=features)
            st.dataframe(coefs.rename("Coefficient"))

        # Download results
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Download Results CSV",
            data=csv,
            file_name="predicted_results.csv",
            mime="text/csv"
        )
