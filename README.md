Revolution Risk Predictor - Machine Learning Project
📋 Executive Summary

The Revolution Risk Predictor is a machine learning solution designed to analyze socioeconomic and political indicators to predict the likelihood of revolutionary events in countries. This tool provides actionable insights for policymakers, NGOs, and international organizations by identifying countries at risk of political instability.

Our solution employs two machine learning models (Logistic Regression and Random Forest) to analyze key risk factors and generates risk assessments with interpretable results. The system achieved strong performance metrics with a PR-AUC of 0.85+ on test data, demonstrating its effectiveness in identifying at-risk countries.
🎯 Problem Statement

Political instability and revolutionary events have significant humanitarian and economic consequences. Traditional analysis methods often fail to:

    Process multiple socioeconomic indicators simultaneously

    Provide quantitative risk assessments

    Offer timely predictions for preventive action

This project addresses these limitations by developing a machine learning system that:

    Analyzes multiple risk factors concurrently

    Provides quantifiable risk scores (0-1 scale)

    Offers monthly updated predictions

    Delivers interpretable results for decision-makers

📊 Dataset Source

Primary Dataset: Synthetic data generated based on real-world indicators from:

    World Bank Development Indicators (GDP, unemployment, demographics)

    Polity5 Project (regime type scores)

    International Telecommunication Union (internet penetration rates)

    Historical revolutionary event patterns

Dataset Characteristics:

    30 countries across multiple continents

    36 months of data (January 2021 - December 2023)

    6 key features per country-month

    Balanced representation of developed and developing nations

Key Features:

    GDP per capita (log-transformed)

    Unemployment rate (%)

    Youth population percentage (%)

    Internet penetration rate (%)

    Polity score (-10 to 10 autocracy-democracy scale)

    Previous revolutionary events (binary)

🧮 Methodology

1. Data Preprocessing

   Time-based train-test split (last 12 months for testing)

   Feature engineering: log transformation of GDP, lag features

   Standardization of numerical features

   Handling of class imbalance using resampling techniques

2. Model Selection

We implemented and compared two machine learning approaches:

Logistic Regression:

    Provides interpretable coefficients

    Fast training and prediction

    Good baseline performance

Random Forest:

    Handles non-linear relationships

    Robust to outliers

    Provides feature importance rankings

3. Evaluation Metrics

Given the imbalanced nature of revolutionary event data, we prioritized:

    PR-AUC (Primary metric): Precision-Recall Area Under Curve

    ROC-AUC: Receiver Operating Characteristic Area Under Curve

    F1-Score: Balance of precision and recall

    Brier Score: Calibration of probability estimates

4. Model Interpretation

   Feature importance analysis

   Coefficient interpretation for logistic model

   Error analysis and pattern identification

📈 Results
Performance Comparison
Model Accuracy Precision Recall F1-Score ROC-AUC PR-AUC
Logistic Regression 0.82 0.78 0.75 0.76 0.89 0.83
Random Forest 0.85 0.81 0.79 0.80 0.92 0.86
Key Findings

    Random Forest performed slightly better across all metrics

    Top predictive features: Polity score, GDP, and previous events

    Risk threshold: 0.3+ probability indicates elevated revolution risk

    Model calibration: Well-calibrated probabilities (Brier score: 0.11)

Country Risk Assessment Examples

    High-risk profile: Low GDP, high unemployment, autocratic regime → 0.87 risk score

    Low-risk profile: High GDP, democratic regime, no previous events → 0.12 risk score

    Malaysia: Moderate risk profile → 0.38 risk score

🚀 Demonstration of the Application
How to Run the Project

    Install dependencies:
    bash

pip install -r requirements.txt

Run the complete pipeline:
bash

# Execute notebooks in order:

jupyter notebook notebooks/01_data_preprocessing.ipynb
jupyter notebook notebooks/02_model_development.ipynb
jupyter notebook notebooks/03_model_testing.ipynb

Launch the Streamlit app:
bash

streamlit run src/streamlit_app.py

Application Features

    Interactive Dashboard:

        Real-time risk predictions for all countries

        Adjustable risk threshold slider

        Model comparison and performance metrics

    Custom Scenario Testing:

        Input custom country parameters

        Get instant risk assessments

        Compare different policy scenarios

    Data Visualization:

        Feature importance charts

        Risk distribution histograms

        Model performance comparisons

    Export Capabilities:

        Download prediction results as CSV

        Save visualizations for reporting

Sample Use Case

A policy analyst can:

    Adjust economic indicators to simulate policy interventions

    Compare risk profiles across different countries

    Identify which factors most influence revolution risk

    Export results for inclusion in policy briefs

🔮 Future Enhancements

    Real Data Integration: Connect to live data sources (World Bank API, ACLED)

    Additional Models: Experiment with gradient boosting and neural networks

    Temporal Features: Incorporate time-series analysis for better predictions

    Regional Analysis: Add geographic and regional risk factors

    Early Warning System: Develop alert mechanisms for high-risk countries

📚 Acknowledgments

This project was developed as part of our Machine Learning course requirements. We would like to acknowledge:

    Our Lecturer: For guidance on machine learning principles and evaluation methodologies

    Open Data Providers: World Bank, Polity Project, and other organizations making data publicly available

    Python Community: For maintaining the excellent data science ecosystem (pandas, scikit-learn, Streamlit)

    Team Members: For collaborative effort in developing and testing this solution
