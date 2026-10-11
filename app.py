import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------
# Load model and feature columns
# --------------------------------

model = joblib.load("rossmann_xgboost_model.pkl")
feature_columns = joblib.load("rossmann_features.pkl")


# --------------------------------
# Load data
# --------------------------------

@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    store = pd.read_csv("store.csv")

    df = train.merge(store, on="Store", how="left")

    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"])

    return df


df = load_data()


# --------------------------------
# Prediction function
# --------------------------------

def forecast_store_sales(
    store_id,
    forecast_date,
    promo,
    schoolholiday
):

    history = df[
        (df["store"] == store_id) &
        (df["open"] == 1)
    ].copy()

    history = history.sort_values("date")

    forecast_date = pd.Timestamp(forecast_date)

    if history.empty:
        raise ValueError(f"No historical data found for store {store_id}")

    latest_date = history["date"].max()

    if forecast_date <= latest_date:
        raise ValueError(
            f"Forecast date must be after the latest historical date "
            f"({latest_date.date()})."
        )
    

    latest = history.iloc[-1]

    future = pd.DataFrame({
        "store": [store_id],
        "date": [pd.Timestamp(forecast_date)],
        "promo": [promo],
        "schoolholiday": [schoolholiday],
        "competitiondistance": [latest["competitiondistance"]],
        "promo2": [latest["promo2"]],
        "storetype": [latest["storetype"]],
        "assortment": [latest["assortment"]],
        "stateholiday": [latest["stateholiday"]]
    })

    # Calendar features

    future["year"] = future["date"].dt.year
    future["month"] = future["date"].dt.month
    future["day"] = future["date"].dt.day
    future["dayofweek"] = future["date"].dt.dayofweek
    future["weekofyear"] = (
        future["date"].dt.isocalendar().week.astype(int)
    )
    future["quarter"] = future["date"].dt.quarter
    future["is_weekend"] = (
        future["dayofweek"] >= 5
    ).astype(int)

    # Lag features

    future["lag_1"] = history["sales"].iloc[-1]
    future["lag_7"] = history["sales"].iloc[-7]
    future["lag_14"] = history["sales"].iloc[-14]
    future["lag_28"] = history["sales"].iloc[-28]

    # Rolling features

    future["rolling_mean_7"] = (
        history["sales"].iloc[-7:].mean()
    )

    future["rolling_mean_14"] = (
        history["sales"].iloc[-14:].mean()
    )

    future["rolling_mean_28"] = (
        history["sales"].iloc[-28:].mean()
    )

    future["rolling_std_7"] = (
        history["sales"].iloc[-7:].std()
    )

    future["rolling_std_28"] = (
        history["sales"].iloc[-28:].std()
    )

    # Ratio features

    future["trend_7_28"] = (
        future["rolling_mean_7"] /
        (future["rolling_mean_28"] + 1)
    )

    future["lag1_vs_lag7"] = (
        future["lag_1"] /
        (future["lag_7"] + 1)
    )

    # One-hot encoding

    future = pd.get_dummies(
        future,
        columns=[
            "storetype",
            "assortment",
            "stateholiday"
        ],
        dtype=int
    )

    # Match training features

    X_future = future.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(X_future)[0]

    return prediction



# --------------------------------
# Professional Streamlit Dashboard
# --------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #17365D, #2878B5);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #EAF3FF;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
    }

    .result {
        padding: 1.3rem;
        border-radius: 12px;
        background: var(--secondary-background-color);
        border: 1px solid #2878B5;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class="hero">
    <h1>📈 Rossmann Sales Forecasting</h1>
    <p>
        Predict store sales using historical sales patterns,
        promotions and calendar information.
    </p>
</div>
""", unsafe_allow_html=True)


# Sidebar inputs
st.sidebar.header("Forecast Configuration")
st.sidebar.caption("Choose the store and forecast conditions.")

store_ids = sorted(df["store"].unique().tolist())

with st.sidebar.form("forecast_form"):

    store_id = st.selectbox(
        "Store ID",
        options=store_ids,
        index=0
    )

    forecast_date = st.date_input(
        "Forecast Date",
        value=pd.Timestamp("2015-08-01").date()
    )

    promo = st.selectbox(
        "Promotion",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    schoolholiday = st.selectbox(
        "School Holiday",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    submitted = st.form_submit_button(
        "Predict Sales",
        type="primary",
        use_container_width=True
    )


# Main dashboard
st.subheader("Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric("Test MAE", "723.88")
m2.metric("Test RMSE", "1,070.59")
m3.metric("Test R²", "0.8776")

st.caption(
    "Metrics from the held-out test set. "
    "They describe model evaluation, not the accuracy of an individual forecast."
)

st.subheader("How to Read the Model Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Mean Absolute Error (MAE)**

    The average absolute difference between actual and predicted
    sales was approximately 724 sales units.
    """)

    st.markdown("""
    **Root Mean Squared Error (RMSE)**

    RMSE penalizes large prediction errors more heavily than MAE.
    """)

with col2:
    st.markdown("""
    **R² Score**

    The model achieved an R² score of approximately 0.878 on the
    held-out test set.
    """)

    st.info(
        "These are historical test-set results. Actual forecast "
        "accuracy may differ across stores and dates."
    )

# --------------------------------
# Feature Importance
# --------------------------------

st.divider()
st.subheader("Top 10 Sales Predictors")

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=True
).tail(10)

st.bar_chart(
    importance_df.set_index("Feature"),
    horizontal=True
)

st.caption(
    "Feature importance measures the model's predictive reliance "
    "on each feature; it does not establish causation."
)

st.subheader("Business Insights")

st.markdown("""
- **Recent sales history:** Rolling averages are important
  predictors of future sales.
- **Promotions:** Promotional activity is useful for forecasting
  store demand.
- **Weekly patterns:** Day-of-week and weekend effects help capture
  differences in shopping behavior.
- **Forecast uncertainty:** Individual predictions can still have
  substantial errors, particularly for certain stores and days.
""")

st.divider()

st.subheader("Sales Forecast")
st.write(
    "Enter the forecast conditions in the sidebar and select "
    "**Predict Sales** to generate a prediction."
)


if submitted:

    try:
        prediction = forecast_store_sales(
            store_id=int(store_id),
            forecast_date=forecast_date,
            promo=promo,
            schoolholiday=schoolholiday
        )

        st.markdown(
            f"""
            <div class="result">
                <p>Predicted sales · Store {store_id}</p>
                <h1>{prediction:,.2f}</h1>
                <p>Forecast date: {forecast_date}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Selected Store", str(store_id))
        col2.metric("Promotion", "Yes" if promo else "No")
        col3.metric(
            "School Holiday",
            "Yes" if schoolholiday else "No"
        )

        st.info(
            "This is a model estimate, not a guaranteed sales figure. "
            "Forecast reliability depends on the input features and "
            "whether the forecast period is represented by the training data."
        )

    except (ValueError, IndexError, KeyError) as e:
        st.error(f"Could not generate the forecast: {e}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.divider()

st.caption(
    "Built with Python, Pandas, XGBoost and Streamlit."
)