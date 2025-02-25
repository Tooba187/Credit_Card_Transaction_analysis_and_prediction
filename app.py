import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime

# ---- Page Configuration ----
st.set_page_config(
    page_title="AI-Powered Credit Card Spending Forecast",
    page_icon="💳",
    layout="wide"
)

# ---- Load Dataset ----
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\Data Science\data world ds\task 4\Data\credit_card_transactions_cleaned.csv")  # Update with dataset path
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    return df

df = load_data()

# ---- Sidebar - User Inputs ----
st.sidebar.image("logo.png", width=150)  # Add your logo here
st.sidebar.header("🔍 User Selections")

customer_id = st.sidebar.text_input("Enter Customer ID (optional):")
category = st.sidebar.selectbox("Select Spending Category:", df["category"].unique())
time_range = st.sidebar.slider("Select Time Range (Days):", 30, 365, 180)

# Filter Data Based on Inputs
filtered_df = df[(df['category'] == category) & 
                 (df['trans_date_trans_time'] >= (df['trans_date_trans_time'].max() - pd.Timedelta(days=time_range)))]

# ---- Main Section - Spending Overview ----
st.title("💳 AI-Powered Credit Card Spending Forecast")
st.markdown("**Predict and analyze spending trends using advanced AI & Machine Learning models.**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Recent Transactions")
    st.write(filtered_df[['trans_date_trans_time', 'amt', 'merchant']].head())

with col2:
    st.subheader("📈 Spending Insights")
    total_spent = filtered_df['amt'].sum()
    avg_spent = filtered_df['amt'].mean()
    highest_trans = filtered_df['amt'].max()
    
    st.metric(label="💰 Total Spent", value=f"${total_spent:,.2f}")
    st.metric(label="📉 Avg Transaction", value=f"${avg_spent:,.2f}")
    st.metric(label="🚀 Highest Transaction", value=f"${highest_trans:,.2f}")

# ---- Aggregated Data for Prophet Model ----
daily_spending = filtered_df.groupby(filtered_df['trans_date_trans_time'].dt.date)['amt'].sum().reset_index()
daily_spending.columns = ['ds', 'y']

# ---- Machine Learning - Facebook Prophet ----
if not daily_spending.empty:
    model = Prophet()
    model.fit(daily_spending)

    future = model.make_future_dataframe(periods=180)  # Forecast for 6 Months
    forecast = model.predict(future)

    # ---- Visualization ----
    st.subheader("🔮 Future Spending Forecast")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig)

    # ---- Detailed Forecast Data ----
    with st.expander("📜 View Detailed Forecast Data"):
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

    # ---- Download Forecast Data ----
    forecast_csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast Report", data=forecast_csv, file_name="spending_forecast.csv", mime="text/csv")

else:
    st.warning("⚠️ Not enough data for forecasting. Try increasing the time range!")

# ---- Fraud Detection & Alerts (Optional Feature) ----
fraud_df = filtered_df[filtered_df['is_fraud'] == 1]
if not fraud_df.empty:
    st.sidebar.error(f"🚨 **Fraudulent Transactions Detected: {len(fraud_df)}**")
    st.sidebar.write(fraud_df[['trans_date_trans_time', 'amt', 'merchant']])

