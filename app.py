import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pygam import LinearGAM, s
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from scipy.stats import normaltest
import matplotlib.pyplot as plt
import seaborn as sns
import math
import io
from datetime import datetime

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def prepare_data(df):
    X = df.drop(columns=excluded_cols, errors='ignore')
    y = df['occupancy_rate']
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    return X.loc[valid_idx], y.loc[valid_idx]

# columns excluded from modeling input
excluded_cols = [
    'scode', 'month_of_dtmonth', 'occ', 'total',
    'net_rentable_square_feet', 'unit_count', 'ami', 
    'asset_value', 'market_cap_rate', 'inventory_units',
    'under_construction_units', 'market_asking_rent/unit',
    'market_effective_rent_growth', 'market_sale_price_per_unit',
    'unemployment_rate', 'building_age', 'occupancy_rate',
    'occupancy_rate_lag2', 'occupancy_rate_lag3', 'occupancy_rate_diff',
    'occupancy_rate_lag1_logit', 'occupancy_rate_lag2_logit',
    'occupancy_rate_lag3_logit', 'occupancy_rate_diff_logit', 'year'
]

def run_linear_regression(df):
    # Define features and target
    X = df.drop(columns=excluded_cols, errors='ignore')
    y = df['occupancy_rate']

    # Handle inf/NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Coefficients
    coefficients = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False).apply(lambda x: format(x, '.6f'))

    # Evaluation report
    report = pd.DataFrame({
        "Metric": ["R²", "MAE", "MSE", "RMSE"],
        "Training": [
            r2_score(y_train, y_pred_train),
            mean_absolute_error(y_train, y_pred_train),
            mean_squared_error(y_train, y_pred_train),
            root_mean_squared_error(y_train, y_pred_train)
        ],
        "Testing": [
            r2_score(y_test, y_pred_test),
            mean_absolute_error(y_test, y_pred_test),
            mean_squared_error(y_test, y_pred_test),
            root_mean_squared_error(y_test, y_pred_test)
        ]
    })

    # P-values
    X_train_sm = sm.add_constant(X_train)
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    p_values = model_sm.pvalues
    significance = []
    for var, p_val in p_values.items():
        if p_val < 0.001:
            significance.append(f"{var:<30} {p_val:.6f} ***")
        elif p_val < 0.01:
            significance.append(f"{var:<30} {p_val:.6f} **")
        elif p_val < 0.05:
            significance.append(f"{var:<30} {p_val:.6f} *")
        elif p_val < 0.1:
            significance.append(f"{var:<30} {p_val:.6f} .")
        else:
            significance.append(f"{var:<30} {p_val:.6f}")

    return coefficients, report, significance, X_test, y_test, y_pred_test


def run_random_forest(df):
    # Feature selection
    X = df.drop(columns=excluded_cols, errors='ignore')
    y = df['occupancy_rate']

    # Handle infs/NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluation
    report = pd.DataFrame({
        "Metric": ["R²", "MAE", "MSE", "RMSE"],
        "Training": [
            r2_score(y_train, y_pred_train),
            mean_absolute_error(y_train, y_pred_train),
            mean_squared_error(y_train, y_pred_train),
            root_mean_squared_error(y_train, y_pred_train)
        ],
        "Testing": [
            r2_score(y_test, y_pred_test),
            mean_absolute_error(y_test, y_pred_test),
            mean_squared_error(y_test, y_pred_test),
            root_mean_squared_error(y_test, y_pred_test)
        ]
    })

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

    return report, importances, X_test, y_test, y_pred_test

def run_xgboost(df):
    # Feature selection
    X = df.drop(columns=excluded_cols, errors='ignore')
    y = df['occupancy_rate']

    # Handle infs/NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluation
    report = pd.DataFrame({
        "Metric": ["R²", "MAE", "MSE", "RMSE"],
        "Training": [
            r2_score(y_train, y_pred_train),
            mean_absolute_error(y_train, y_pred_train),
            mean_squared_error(y_train, y_pred_train),
            root_mean_squared_error(y_train, y_pred_train)
            
        ],
        "Testing": [
            r2_score(y_test, y_pred_test),
            mean_absolute_error(y_test, y_pred_test),
            mean_squared_error(y_test, y_pred_test),
            root_mean_squared_error(y_test, y_pred_test)
        ]
    })

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

    return report, importances, X_test, y_test, y_pred_test


def run_gam(df):
    # Step 1: Select features
    X = df.drop(columns=excluded_cols, errors='ignore')
    y = df['occupancy_rate']

    # Step 2: Clean NaNs/Infs
    df_gam = X.copy()
    df_gam['occupancy_rate'] = y
    df_gam = df_gam.replace([np.inf, -np.inf], np.nan).dropna()
    X_clean = df_gam.drop(columns='occupancy_rate')
    y_clean = df_gam['occupancy_rate']
    selected_cols = X_clean.columns.tolist()

    # Step 3: Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    # Step 4: Define GAM model
    n_splines = 10
    terms = reduce(lambda a, b: a + b, [s(i, n_splines=n_splines) for i in range(X_train.shape[1])])
    gam = LinearGAM(terms)
    gam.gridsearch(X_train.values.astype(float), y_train.values.astype(float))

    # Step 5: Predict & Evaluate
    y_pred_train = gam.predict(X_train.values.astype(float))
    y_pred_test = gam.predict(X_test.values.astype(float))

    report = pd.DataFrame({
        "Metric": ["R²", "MAE", "MSE", "RMSE"],
        "Training": [
            r2_score(y_train, y_pred_train),
            mean_absolute_error(y_train, y_pred_train),
            mean_squared_error(y_train, y_pred_train),
            root_mean_squared_error(y_train, y_pred_train)
        ],
        "Testing": [
            r2_score(y_test, y_pred_test),
            mean_absolute_error(y_test, y_pred_test),
            mean_squared_error(y_test, y_pred_test),
            root_mean_squared_error(y_test, y_pred_test)
        ]
    })

    # Step 6: Plot partial dependence
    fig_pd, axes = plt.subplots(math.ceil(len(selected_cols) / 3), 3, figsize=(18, 5 * math.ceil(len(selected_cols)/3)))
    axes = axes.flatten()
    for i, term in enumerate(selected_cols):
        XX = gam.generate_X_grid(term=i)
        pd_mean, pd_ci = gam.partial_dependence(term=i, X=XX, width=0.95)
        axes[i].plot(XX[:, i], pd_mean, label='Partial Effect')
        axes[i].plot(XX[:, i], pd_ci[:, 0], 'r--', label='95% CI' if i == 0 else "")
        axes[i].plot(XX[:, i], pd_ci[:, 1], 'r--')
        axes[i].set_title(f'{term} vs. occupancy_rate')
        axes[i].set_xlabel(term)
        axes[i].set_ylabel('Partial Effect')
        axes[i].grid(True)
    plt.tight_layout()
    plt.close(fig_pd)
    return report, fig_pd, X_test, y_test, y_pred_test

def run_all_models(df):
    results = []

    # Linear Regression
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append(pd.DataFrame({
        "Model": "Linear Regression",
        "Actual": y_test.values,
        "Predicted": y_pred
    }))

    # Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append(pd.DataFrame({
        "Model": "Random Forest",
        "Actual": y_test.values,
        "Predicted": y_pred
    }))

    # XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append(pd.DataFrame({
        "Model": "XGBoost",
        "Actual": y_test.values,
        "Predicted": y_pred
    }))

    # GAM
    from pygam import LinearGAM, s
    from functools import reduce
    terms = reduce(lambda a, b: a + b, [s(i, n_splines=10) for i in range(X_train.shape[1])])
    gam = LinearGAM(terms)
    gam.gridsearch(X_train.values.astype(float), y_train.values.astype(float))
    y_pred = gam.predict(X_test.values.astype(float))
    results.append(pd.DataFrame({
        "Model": "GAM",
        "Actual": y_test.values.astype(float),
        "Predicted": y_pred
    }))

    return pd.concat(results, ignore_index=True)


def make_prediction_df(model_name, X_test, y_test, y_pred):
    return pd.DataFrame({
        "Model": model_name,
        "Actual": y_test.values.astype(float),
        "Predicted": y_pred
    })


def export_results_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    return output.getvalue()

def export_results_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


filename = f"occupancy_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

# --- Streamlit App ---
st.title("Occupancy Rate Multi-Model Prediction Platform")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    mode = None
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())
    
    mode = st.radio("Select Mode", ["Run Single Model", "Run All Models"])

    if mode == "Run Single Model":
        
        selected_model = st.selectbox("Select model to run", ["Linear Regression", "Random Forest", "XGBoost", "GAM"])

        if st.button("Run Selected Model"):
            
            with st.spinner(f"Training {selected_model} model..."):
                if selected_model == "Linear Regression":
                    coefs, metrics_report, pval_output, X_test, y_test, y_pred_test = run_linear_regression(df)
                    st.subheader("Coefficients")
                    st.dataframe(coefs)
                    st.subheader("Evaluation")
                    st.dataframe(metrics_report)
                    st.subheader("P-values")
                    for line in pval_output:
                        st.text(line)
                    single_model_df = make_prediction_df("Linear Regression", X_test, y_test, y_pred_test)
                    csv_data = export_results_to_csv(single_model_df)

                    st.download_button(
                        label="Download Linear Regression Predictions (CSV)",
                        data=csv_data,
                        file_name=f"linear_regression_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                        )
                elif selected_model == "Random Forest":
                    rf_report, rf_importances, X_test, y_test, y_pred_test = run_random_forest(df)
                    st.dataframe(rf_report)
                    st.bar_chart(rf_importances.sort_values(ascending=False))
                    single_model_df = make_prediction_df("Random Forest", X_test, y_test, y_pred_test)
                    csv_data = export_results_to_csv(single_model_df)

                    st.download_button(
                        label="Download Random Forest Predictions (CSV)",
                        data=csv_data,
                        file_name=f"random_forest_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                        )

                elif selected_model == "XGBoost":
                    xgb_report, xgb_importances, X_test, y_test, y_pred_test = run_xgboost(df)
                    st.dataframe(xgb_report)
                    st.bar_chart(xgb_importances.sort_values(ascending=False))
                    single_model_df = make_prediction_df("XGBoost", X_test, y_test, y_pred_test)
                    csv_data = export_results_to_csv(single_model_df)

                    st.download_button(
                        label="Download XGBoost Predictions (CSV)",
                        data=csv_data,
                        file_name=f"xgboost_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                        )

                elif selected_model == "GAM":
                    gam_report, fig_partial, X_test, y_test, y_pred_test = run_gam(df)
                    st.dataframe(gam_report)
                    st.pyplot(fig_partial)
                    single_model_df = make_prediction_df("GAM", X_test, y_test, y_pred_test)
                    csv_data = export_results_to_csv(single_model_df)

                    st.download_button(
                        label="Download GAM Predictions (CSV)",
                        data=csv_data,
                        file_name=f"gam_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                        )
                
    elif mode == "Run All Models":
        if st.button("Run All Models and Export"):
            final_df = run_all_models(df)
            st.dataframe(final_df)

            csv_data = export_results_to_csv(final_df)
            st.download_button(
            label="Download All Model Predictions (CSV)",
            data=csv_data,
            file_name=f"all_model_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
)

else:
    st.info("Please upload a CSV file to start.")
# --- Streamlit App ---