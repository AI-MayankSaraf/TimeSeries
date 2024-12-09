import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Configure Page
st.set_page_config(page_title="Time Series App", layout="wide")

# Title
st.title("📊 Time Series Analysis App")

# File Upload
st.sidebar.header("Upload Time Series Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file, parse_dates=True)
    st.write("### Uploaded Data", df.head())

    # Data Preprocessing
    st.sidebar.subheader("Preprocessing Options")
    date_column = st.sidebar.selectbox("Select Date Column", df.columns[0])
    value_column = st.sidebar.selectbox("Select Value Column", df.columns[1])

    # Validate Uploaded Data
    if "ds" not in df.columns or "y" not in df.columns:
        st.sidebar.warning("Ensure your CSV contains the selected columns.")


    if date_column and value_column:
        df = df[[date_column, value_column]].rename(
            columns={date_column: "ds", value_column: "y"}
        )
    else:
        st.error("Please select valid columns for Date and Value.")
            
    # Verify renaming
    st.write("### Processed Data", df.head())

    # Convert `ds` to datetime
    try:
        df["ds"] = pd.to_datetime(df["ds"], errors='coerce')  # Handle invalid dates
        invalid_dates = df[df["ds"].isna()]
        
        if not invalid_dates.empty:
            st.warning("Some rows have invalid dates and were removed.")
            st.write("### Invalid Date Rows", invalid_dates)

        # Drop rows with invalid dates
        df = df.dropna(subset=["ds"])

    except Exception as e:
        st.error(f"Error in converting Date column: {e}")
    
    # Handle Missing Values
    if st.sidebar.checkbox("Fill Missing Values"):
        method = st.sidebar.selectbox("Filling Method", ["Forward Fill", "Backward Fill", "Interpolate"])
        if method == "Forward Fill":
            df["y"] = df["y"].fillna(method="ffill")
        elif method == "Backward Fill":
            df["y"] = df["y"].fillna(method="bfill")
        else:
            df["y"] = df["y"].interpolate()

    # Resampling Data
    if st.sidebar.checkbox("Resample Data"):
        freq = st.sidebar.selectbox("Frequency", ["D", "W", "M"], index=0)
        df = df.resample(freq, on="ds").mean().reset_index()

    # Date Range Selection
    date_range = st.sidebar.date_input("Select Date Range", [df["ds"].min(), df["ds"].max()])
    df = df[(df["ds"] >= pd.Timestamp(date_range[0])) & (df["ds"] <= pd.Timestamp(date_range[1]))]

    # Visualization Options
    st.write("### Time Series Visualization")
    chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Scatter"])
    if chart_type == "Line":
        fig = px.line(df, x="ds", y="y", title="Time Series Data")
    elif chart_type == "Bar":
        fig = px.bar(df, x="ds", y="y", title="Time Series Data")
    else:
        fig = px.scatter(df, x="ds", y="y", title="Time Series Data")
    st.plotly_chart(fig, use_container_width=True)

    # Seasonal Decomposition
    if st.sidebar.checkbox("Show Seasonal Decomposition"):
        st.write("### Seasonal Decomposition")
        decomposition = seasonal_decompose(df.set_index("ds")["y"], model="additive", period=30)
        st.write("#### Trend")
        st.line_chart(decomposition.trend)
        st.write("#### Seasonality")
        st.line_chart(decomposition.seasonal)
        st.write("#### Residual")
        st.line_chart(decomposition.resid)

    # Forecasting
    st.sidebar.subheader("Forecasting Options")
    forecast_period = st.sidebar.slider("Forecast Period (Days)", 1, 365, 30)

    if st.sidebar.button("Run Forecasting"):
        st.write("### Forecasting Results")
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_period)
        forecast = model.predict(future)

        # Plot forecast
        forecast_fig = model.plot(forecast)
        st.pyplot(forecast_fig)

        # Performance Metrics
        st.write("### Model Performance Metrics")
        actual = df["y"].values
        predicted = forecast["yhat"].iloc[:len(actual)].values
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = math.sqrt(mse)
        st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Export Results
        csv = forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Forecast",
            data=csv,
            file_name="forecast.csv",
            mime="text/csv",
        )

else:
    st.info("Please upload a CSV file to start.")
