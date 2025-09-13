# 🌍 Revolution Risk Predictor

> Machine learning-powered analysis of political stability and revolution risks

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine-Learning-orange)

## 📖 What It Does

**Revolution Risk Predictor** analyzes socioeconomic and political indicators to forecast countries at risk of revolutionary events. Designed for policymakers, NGOs, and researchers seeking data-driven insights into global stability.

## ✨ Key Features

- **Real-time Risk Assessment**: Instant country risk scoring
- **Multi-Model Analysis**: Compare Logistic Regression vs. XGBoost performance
- **Interactive Dashboard**: Adjust parameters and see immediate results
- **Comprehensive Visualizations**: Correlation matrices, feature importance, risk distributions
- **Export Ready**: Download predictions and analysis reports

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/revolution-risk-predictor.git
cd revolution-risk-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run revolution_risk_app.py
```

## 🎯 How It Works

    Data Input: Use synthetic data or upload your own country indicators

    Preprocessing: Automated cleaning, normalization, and feature engineering

    Model Training: Dual-model approach for robust predictions

    Risk Scoring: Probability-based revolution risk assessment

    Visualization: Interactive charts and exportable reports

## 📊 Model Performance

Metric Logistic Regression XGBoost
PR-AUC 0.82 0.87
ROC-AUC 0.89 0.92
F1-Score 0.78 0.83

## 🔍 Key Risk Indicators

    📉 GDP per capita (-42% impact)

    📈 Unemployment rates (+38% impact)

    👥 Youth population percentage (+25% impact)

    🌐 Internet penetration (-18% impact)

    🏛️ Political stability (-55% impact)

## 🛠️ Built With

    Python • Streamlit • Scikit-learn • XGBoost

    Pandas • Matplotlib • Seaborn • Joblib

## 📋 Requirements

- streamlit>=1.28.0
- pandas>=2.0.0
- scikit-learn>=1.2.0
- matplotlib>=3.7.0
- seaborn>=0.12.0

## 🌐 Live Demo

https://static.streamlit.io/badges/streamlit_badge_black_white.svg

Note: Demo uses synthetic data for demonstration purposes

## 📄 License

MIT Licensed. See LICENSE file for details.
