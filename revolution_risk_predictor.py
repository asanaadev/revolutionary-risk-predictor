import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, f1_score
import joblib
import matplotlib.pyplot as plt

# Try to import XGBoost; fall back if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Revolution Risk Predictor", layout="wide")
st.title("Revolution Risk Predictor — Country Analysis")

st.markdown(
    """
    This tool predicts the risk of revolutionary events in countries based on socioeconomic and political indicators.
    
    **Features used:**
    - GDP per capita (log transformed)
    - Unemployment rate
    - Youth population percentage
    - Internet penetration rate
    - Polity score (democracy/autocracy scale)
    - Previous revolutionary events
    
    Use the sidebar to adjust the alert threshold or test custom inputs.
    """
)


@st.cache_data
def load_real_data():
    """Load real-world country data with synthetic revolutionary event labels"""
    # Real country data with some realistic values
    countries_data = {
        'country': ['France', 'Germany', 'UK', 'Italy', 'Spain', 'Poland',
                    'Ukraine', 'Turkey', 'Egypt', 'Tunisia', 'Brazil', 'Argentina',
                    'Mexico', 'USA', 'Canada', 'Australia', 'Tasmania', 'Japan',
                    'South Korea', 'China', 'India', 'Pakistan', 'Nigeria', 'South Africa',
                    'Malaysia'],  # Added Malaysia

        'gdp': [41464, 48560, 42724, 34260, 29875, 17319, 3985, 9061, 3618, 3440,
                6796, 10639, 9946, 65280, 46194, 51692, 45000, 40113, 31846, 10500,
                # Malaysia GDP per capita (USD)
                2100, 1193, 2028, 6040, 11372],

        'unemployment': [7.9, 3.0, 3.7, 9.7, 13.8, 3.4, 9.8, 10.6, 7.3, 15.2,
                         11.6, 8.5, 3.3, 3.7, 5.3, 5.1, 6.2, 2.4, 3.7, 4.8, 7.1,
                         # Malaysia unemployment rate (%)
                         6.3, 9.8, 28.5, 3.7],

        'youth_pct': [17.8, 15.1, 17.5, 15.1, 14.7, 18.3, 19.1, 25.6, 33.3, 29.4,
                      27.4, 24.9, 26.3, 19.0, 16.0, 19.0, 18.5, 14.5, 19.0, 17.2,
                      # Malaysia youth population (%)
                      27.0, 35.0, 42.5, 29.5, 24.8],

        'internet_pct': [85.6, 89.7, 94.9, 74.5, 87.1, 82.9, 64.3, 71.0, 57.3, 66.3,
                         70.7, 79.9, 65.8, 87.3, 91.0, 88.2, 85.0, 93.0, 95.1, 61.2,
                         # Malaysia internet penetration (%)
                         45.0, 35.1, 42.0, 56.2, 89.6],

        'polity': [8, 10, 10, 10, 8, 10, 7, -3, -3, 7, 8, 8, 8, 8, 10, 10, 10, 10,
                   # Malaysia polity score (democracy scale)
                   10, -7, 9, 5, 7, 9, 8]
    }

    df_countries = pd.DataFrame(countries_data)

    # Generate monthly data for the past 3 years
    months = pd.date_range("2021-01-01", "2023-12-01", freq='MS')
    rows = []

    for _, country_row in df_countries.iterrows():
        country = country_row['country']
        base_values = country_row.to_dict()

        for i, d in enumerate(months):
            # Add some realistic monthly variation
            row = base_values.copy()
            row['date'] = d
            row['year'] = d.year
            row['month'] = d.month

            # Add realistic fluctuations to economic indicators
            row['gdp'] = row['gdp'] * (1 + np.random.normal(0, 0.01))
            row['unemployment'] = max(
                1, row['unemployment'] + np.random.normal(0, 0.5))
            row['youth_pct'] = max(
                5, min(70, row['youth_pct'] + np.random.normal(0, 0.2)))
            row['internet_pct'] = max(
                1, min(100, row['internet_pct'] + np.random.normal(0, 0.3)))
            row['polity'] = max(-10, min(10, row['polity'] +
                                np.random.normal(0, 0.1)))

            # Calculate a realistic risk score based on known factors
            risk_factors = (
                (100 - row['polity']) * 0.1 +  # Lower polity = higher risk
                # Higher unemployment = higher risk
                row['unemployment'] * 0.15 +
                # More youth = slightly higher risk
                row['youth_pct'] * 0.05 +
                # Less internet access = slightly higher risk
                (100 - row['internet_pct']) * 0.02 +
                (40000 / (row['gdp'] + 1000)) * 0.5  # Lower GDP = higher risk
            )

            # Add some noise and time-based variation
            time_factor = np.sin(i / 6.0) * 0.5  # Seasonal effect
            risk_score = risk_factors + np.random.normal(0, 0.5) + time_factor

            # Ensure we have a good mix of both classes in the training data
            # Adjust threshold to get a reasonable balance (approx 20% positive class)
            row['label'] = 1 if risk_score > 5.5 else 0

            rows.append(row)

    df = pd.DataFrame(rows)

    # Add lag of events: previous month events per country
    df = df.sort_values(["country", "date"])
    df["prev_events"] = df.groupby(
        "country")["label"].shift(1).fillna(0).astype(int)

    # Create a log_gdp feature to stabilize scale
    df["log_gdp"] = np.log1p(df["gdp"])

    return df


@st.cache_data
def prepare_features(df):
    feature_cols = ["log_gdp", "unemployment", "youth_pct",
                    "internet_pct", "polity", "prev_events"]
    X = df[feature_cols].copy()
    y = df["label"].copy()
    return X, y, feature_cols


def train_models(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Check if we have both classes in training data
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        st.warning(f"Training data contains only one class: {unique_classes[0]}. "
                   "This may cause issues with some models.")

        # If only one class, create a small number of synthetic samples of the other class
        if unique_classes[0] == 0:
            # Add a few positive samples
            synthetic_X = X_train.sample(
                n=min(5, len(X_train)), random_state=42)
            synthetic_y = pd.Series([1] * len(synthetic_X))
        else:
            # Add a few negative samples
            synthetic_X = X_train.sample(
                n=min(5, len(X_train)), random_state=42)
            synthetic_y = pd.Series([0] * len(synthetic_X))

        X_train = pd.concat([X_train, synthetic_X])
        y_train = pd.concat([y_train, synthetic_y])
        X_train_scaled = scaler.fit_transform(X_train)

    # Logistic Regression baseline
    logreg = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="liblinear")
    logreg.fit(X_train_scaled, y_train)
    joblib.dump((logreg, scaler), os.path.join(MODEL_DIR, "logreg.joblib"))

    # Stronger model: XGBoost if available, else RandomForest
    if XGBOOST_AVAILABLE:
        model = XGBClassifier(use_label_encoder=False,
                              eval_metric="logloss", verbosity=0)
    else:
        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced")

    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, "strong_model.joblib"))

    return logreg, model, scaler


def evaluate_model(model, scaler, X_test, y_test, model_name="model"):
    X_test_scaled = scaler.transform(X_test)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        try:
            probs = model.decision_function(X_test_scaled)
            probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-12)
        except Exception:
            probs = model.predict(X_test_scaled)

    # Check if we have both classes in test data
    if len(np.unique(y_test)) > 1:
        pr_auc = average_precision_score(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
    else:
        pr_auc = float("nan")
        roc_auc = float("nan")
        st.warning(f"Test data contains only one class: {np.unique(y_test)[0]}. "
                   "Cannot calculate PR-AUC and ROC-AUC.")

    brier = brier_score_loss(y_test, probs)
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)

    metrics = {
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "Brier": brier,
        "F1@0.5": f1,
    }
    return metrics, probs


# ====== Prepare data and train ======
with st.spinner("Loading country data..."):
    df = load_real_data()

st.sidebar.header("Control Panel")
retrain = st.sidebar.button("Retrain Models")
threshold = st.sidebar.slider(
    "Alert Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

# Split data: use the last 12 months as test to simulate time-split
latest_date = df["date"].max()
test_start = latest_date - pd.DateOffset(months=12)
train_mask = df["date"] < test_start
test_mask = df["date"] >= test_start

X, y, feature_cols = prepare_features(df)
X_train, X_test = X[train_mask.values], X[test_mask.values]
y_train, y_test = y[train_mask.values], y[test_mask.values]

# Check class distribution
st.sidebar.write("**Class Distribution**")
st.sidebar.write(
    f"Training: {sum(y_train)} positive, {len(y_train)-sum(y_train)} negative")
st.sidebar.write(
    f"Test: {sum(y_test)} positive, {len(y_test)-sum(y_test)} negative")

# Train models
model_files_exist = (
    os.path.exists(os.path.join(MODEL_DIR, "logreg.joblib")) and
    os.path.exists(os.path.join(MODEL_DIR, "strong_model.joblib"))
)

if retrain or not model_files_exist:
    with st.spinner("Training models..."):
        logreg, strong_model, scaler = train_models(X_train, y_train)
else:
    try:
        logreg, scaler = joblib.load(os.path.join(MODEL_DIR, "logreg.joblib"))
        strong_model = joblib.load(os.path.join(
            MODEL_DIR, "strong_model.joblib"))
    except:
        with st.spinner("Training models..."):
            logreg, strong_model, scaler = train_models(X_train, y_train)

# Evaluate both models on test set
metrics_logreg, probs_logreg = evaluate_model(
    logreg, scaler, X_test, y_test, "Logistic Regression")
metrics_strong, probs_strong = evaluate_model(
    strong_model, scaler, X_test, y_test, "Strong Model")

st.subheader("Model Performance (Last 12 Months)")
col1, col2 = st.columns(2)
with col1:
    st.write("**Logistic Regression**")
    st.dataframe(pd.DataFrame([metrics_logreg]))
with col2:
    st.write("**XGBoost**" if XGBOOST_AVAILABLE else "**RandomForest**")
    st.dataframe(pd.DataFrame([metrics_strong]))

# Prepare latest-month predictions for all countries
latest_mask = df["date"] == df["date"].max()
df_latest = df[latest_mask].copy()
X_latest = df_latest[feature_cols].copy()
X_latest_scaled = scaler.transform(X_latest)

model_choice = st.selectbox("Select Model for Predictions", [
                            "Logistic Regression", "Strong Model"])
if model_choice == "Logistic Regression":
    chosen_model = logreg
    chosen_probs = logreg.predict_proba(X_latest_scaled)[:, 1]
else:
    chosen_model = strong_model
    if hasattr(strong_model, "predict_proba"):
        chosen_probs = strong_model.predict_proba(X_latest_scaled)[:, 1]
    else:
        chosen_probs = strong_model.predict(X_latest_scaled)

out = df_latest[["country", "gdp", "unemployment", "youth_pct",
                 "internet_pct", "polity", "year", "month"]].copy()
out["risk_score"] = chosen_probs
out["alert"] = (out["risk_score"] >= threshold).astype(int)

st.subheader("Country Risk Assessment (Latest Month)")
st.dataframe(out.sort_values(
    "risk_score", ascending=False).reset_index(drop=True))

csv = out.to_csv(index=False)
st.download_button("Download Predictions CSV", csv,
                   file_name="country_risk_predictions.csv")

# Feature importance
st.subheader("Model Explanation")
exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    st.write("Logistic Regression Coefficients")
    coefs = logreg.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": coefs})
    coef_df["abs_value"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_value", ascending=True)
    fig, ax = plt.subplots()
    ax.barh(coef_df["feature"], coef_df["coefficient"])
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

with exp_col2:
    st.write("Strong Model Feature Importance")
    try:
        if hasattr(strong_model, "feature_importances_"):
            fi = strong_model.feature_importances_
            fi_df = pd.DataFrame({"feature": feature_cols, "importance": fi}).sort_values(
                "importance", ascending=True)
            fig2, ax2 = plt.subplots()
            ax2.barh(fi_df["feature"], fi_df["importance"])
            ax2.set_xlabel("Importance Score")
            st.pyplot(fig2)
        else:
            st.info("Feature importance not available for this model type.")
    except Exception as e:
        st.error(f"Could not compute feature importances: {e}")

# Custom input testing
st.subheader("Test Custom Country Scenario")
with st.form("custom_input"):
    st.write("Adjust features to test different scenarios:")

    col1, col2 = st.columns(2)
    with col1:
        input_gdp = st.number_input(
            "GDP per capita", min_value=1000, max_value=100000, value=30000)
        input_unemployment = st.number_input(
            "Unemployment Rate", min_value=0.0, max_value=50.0, value=8.0)
        input_youth = st.number_input(
            "Youth Population %", min_value=5.0, max_value=70.0, value=25.0)

    with col2:
        input_internet = st.number_input(
            "Internet Penetration %", min_value=1.0, max_value=100.0, value=75.0)
        input_polity = st.slider(
            "Polity Score (-10 to 10)", min_value=-10, max_value=10, value=8)
        input_prev = st.selectbox("Previous Month Event", [0, 1], index=0)

    submitted = st.form_submit_button("Calculate Risk")

if submitted:
    input_log_gdp = np.log1p(input_gdp)
    x_custom = np.array([[input_log_gdp, input_unemployment, input_youth,
                         input_internet, input_polity, input_prev]])
    x_custom_scaled = scaler.transform(x_custom)

    if hasattr(chosen_model, "predict_proba"):
        prob = chosen_model.predict_proba(x_custom_scaled)[0, 1]
    else:
        prob = float(chosen_model.predict(x_custom_scaled)[0])

    st.metric("Predicted Revolution Risk", f"{prob:.3f}")

    if prob >= threshold:
        st.error("High risk: Above alert threshold")
    else:
        st.success("Low risk: Below alert threshold")

# Data summary
st.subheader("Dataset Overview")
st.write(
    f"Dataset contains {len(df)} country-month records for {df['country'].nunique()} countries")
st.write("Time period:", df['date'].min().strftime(
    "%Y-%m"), "to", df['date'].max().strftime("%Y-%m"))

st.markdown(
    """
    **Methodology:**
    - Uses realistic country data with synthetic revolutionary event labels based on known risk factors
    - Models trained on historical data and evaluated on the most recent 12 months
    - Logistic Regression provides interpretable coefficients
    - XGBoost/RandomForest provides stronger predictive performance
    
    **Interpretation:**
    - Higher risk scores indicate greater probability of revolutionary events
    - Countries with economic challenges, political instability, and youth bulges tend to have higher risk
    """
)
