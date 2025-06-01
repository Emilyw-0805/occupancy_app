import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor # Make sure you import XGBRegressor

# --- Page Configuration ---
st.set_page_config(
    page_title="Occupancy Rate Prediction",
    page_icon="🏠",
    layout="wide"
)

# --- Load Models ---
@st.cache_resource # Cache the model loading for better performance
def load_models():
    try:
        linear_model = joblib.load('linear_regression_model.pkl')
        xgboost_model = joblib.load('xgboost_model.pkl')
        return linear_model, xgboost_model
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'linear_regression_model.pkl' and 'xgboost_model.pkl' are in the same directory.")
        return None, None

linear_model, xgboost_model = load_models()

FEATURE_COLUMNS = [
    "occupancy_rate_lag1", "ami_lag1", "asset_value_lag1", "inventory_units_lag1",
    "under_construction_units_lag1", "market_asking_rent/unit_lag1",
    "market_effective_rent_growth_lag1", "market_sale_price_per_unit_lag1",
    "unemployment_rate_lag1", "market_cap_rate_lag1"
]

# --- App Title and Description ---
st.title("🏡 Occupancy Rate Prediction App")
st.markdown("""
Upload your new property data in CSV format, and I'll predict the **occupancy rate** using both Linear Regression and XGBoost models.
""")

st.write("---")

# --- Upload CSV Section ---
st.header("1. Upload Your Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

df_new_data = None
if uploaded_file is not None:
    try:
        df_new_data = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")
        st.write("Preview of your uploaded data:")
        st.dataframe(df_new_data.head())

        # Check for required columns
        missing_columns = [col for col in FEATURE_COLUMNS if col not in df_new_data.columns]
        if missing_columns:
            st.warning(f"Warning: The uploaded CSV is missing the following required feature columns: {', '.join(missing_columns)}. Please ensure your CSV contains all necessary columns for prediction.")
            df_new_data = None # Invalidate df_new_data if essential columns are missing
        else:
            # Prepare data for prediction: select only the feature columns
            X_new = df_new_data[FEATURE_COLUMNS].copy()

            # Handle inf / NaN values in the new data for prediction
            X_new = X_new.replace([np.inf, -np.inf], np.nan)
            # For NaNs, you'll need to decide on an imputation strategy.
            # For simplicity, we'll just drop rows with NaNs for now, but
            # in a real application, you might want more sophisticated imputation.
            X_new = X_new.dropna()

            if X_new.empty:
                st.error("After handling missing values, no valid rows remain for prediction. Please check your data.")
                df_new_data = None # Invalidate df_new_data
            else:
                st.success(f"Data prepared for prediction. {len(X_new)} rows will be processed.")
                # Store X_new back into df_new_data for easy access later
                df_new_data = X_new

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df_new_data = None

st.write("---")

# --- Prediction Section ---
st.header("2. Make Predictions")

if df_new_data is not None and linear_model is not None and xgboost_model is not None:
    if st.button("Predict Occupancy Rates"):
        st.subheader("Prediction Results:")

        # Ensure that the column order for prediction matches the training order
        X_predict = df_new_data[FEATURE_COLUMNS]

        with st.spinner("Predicting with Linear Regression..."):
            lr_predictions = linear_model.predict(X_predict)
        
        with st.spinner("Predicting with XGBoost..."):
            xgb_predictions = xgboost_model.predict(X_predict)
        
        # Add predictions to the original dataframe (or a copy)
        results_df = df_new_data.copy()
        results_df['Predicted_Occupancy_Rate_Linear_Regression'] = lr_predictions
        results_df['Predicted_Occupancy_Rate_XGBoost'] = xgb_predictions

        st.dataframe(results_df)

        st.download_button(
            label="Download Predictions as CSV",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name="occupancy_predictions.csv",
            mime="text/csv",
        )
else:
    if uploaded_file is None:
        st.info("Please upload a CSV file to make predictions.")
    elif linear_model is None or xgboost_model is None:
        st.warning("Models could not be loaded. Please check the model files.")

st.write("---")

# --- Important Note on Feature Columns ---
st.sidebar.header("📝 Important Note")
st.sidebar.info(
    "Ensure your uploaded CSV contains the **exact same feature columns** "
    "used for training the models. If columns are missing or named differently, "
    "predictions may fail or be inaccurate."
)
st.sidebar.markdown("---")
st.sidebar.write("Developed by UCI MSBA Emily")