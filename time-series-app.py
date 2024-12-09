import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import matplotlib.pyplot as plt

# Title
st.set_page_config(page_title="Time Series Application", layout="wide")
st.title("📈 Advanced Time Series Application")

# File Upload
st.sidebar.header("Upload Time Series Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file, parse_dates=True)
    st.write("### Uploaded Data", df.head())

    # Data Selection
    st.sidebar.subheader("Data Configuration")
    date_column = st.sidebar.selectbox("Select Date Column", df.columns)
    value_column = st.sidebar.selectbox("Select Value Column", df.columns)

    df = df[[date_column, value_column]].rename(
        columns={date_column: "ds", value_column: "y"}
    )

    # Display Time Series Plot
    st.write("### Time Series Plot")
    fig = px.line(df, x="ds", y="y", title="Time Series Data")
    st.plotly_chart(fig, use_container_width=True)

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

        # Show forecast data
        st.write("### Forecast Data", forecast.tail())

    # Export Results
    if st.sidebar.button("Export Forecast"):
        csv = forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Forecast",
            data=csv,
            file_name="forecast.csv",
            mime="text/csv",
        )
else:
    st.info("Please upload a CSV file to proceed.")

