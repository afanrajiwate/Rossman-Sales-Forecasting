# Rossmann Store Sales Forecasting

## Project Overview

This project uses machine learning to forecast retail store sales using historical sales patterns, promotional activity, calendar information, and store characteristics.

The project includes exploratory data analysis, feature engineering, model comparison, XGBoost hyperparameter tuning, and a Streamlit dashboard for generating sales predictions.

## Objectives

* Explore historical retail sales patterns.
* Engineer time-based, lag, and rolling statistical features.
* Compare regression models using a chronological train-test split.
* Tune an XGBoost regression model.
* Build an interactive sales forecasting dashboard.

## Tech Stack

* Python
* Pandas and NumPy
* Matplotlib and Seaborn
* Scikit-learn
* XGBoost
* Streamlit
* Joblib
* Jupyter Notebook

## Model Performance

| Metric | Test Result |
| ------ | ----------: |
| MAE    |      723.88 |
| RMSE   |    1,070.59 |
| R²     |      0.8776 |

These are results from the held-out test set. They do not guarantee the same accuracy for every store or future date.

## Key Findings

* Rolling historical sales averages were important predictive features.
* Promotion and day-of-week information helped the model capture demand patterns.
* Prediction errors varied across stores and days, so individual forecasts should be interpreted with care.

## Application

The Streamlit application allows users to select a store, choose forecast conditions, and view a predicted sales value alongside model performance information.

## Limitations

* The dataset contains historical data, so predictions beyond its time range require careful validation.
* Forecast accuracy can vary across stores and dates.
* Feature importance describes predictive reliance, not causation.
* The forecasting pipeline should use only information available at the forecast time.

## How to Run

1. Install the dependencies:

   `pip install -r requirements.txt`

2. Start the dashboard:

   `python -m streamlit run app.py`

## Project Structure

* `app.py` — Streamlit dashboard
* `Rossmann_Sales_Forecasting.ipynb` — analysis and model development
* `rossmann_xgboost_model.pkl` — saved trained model
* `rossmann_features.pkl` — expected model features
* `requirements.txt` — Python dependencies
* `screenshots/` — dashboard screenshots

## Author

Afan Rajiwate
Github-afanrajiwate