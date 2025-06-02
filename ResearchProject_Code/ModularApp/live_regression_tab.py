import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import pytz

from app_utils import fetch_live_price  # Assumes live BTC/USD data

def run_live_regression_tab():
    """Run the Live Regression tab in the Streamlit app."""
    eastern = pytz.timezone("America/New_York")

    st.subheader("Live Regression Model")
    st.write("Collects live BTC/USD data and forecasts the near-term trend using linear regression.")

    option = st.selectbox(
        "Choose data collection and prediction window:",
        options=[
            ("30s Collection, 15s Forecast", 30, 15),
            ("60s Collection, 20s Forecast", 60, 20),
            ("120s Collection, 30s Forecast", 120, 30),
        ],
        format_func=lambda x: x[0]
    )

    collect_duration, forecast_duration = option[1], option[2]
    st.info(f"Collecting live data for {collect_duration} seconds...")

    live_df = pd.DataFrame(columns=["Time", "Price"])
    start_time = time.time()

    # Data collection loop
    while time.time() - start_time < collect_duration:
        live = fetch_live_price("BTC/USD")
        if live and "timestamp" in live and "last_price" in live:
            timestamp = live["timestamp"].tz_localize("UTC").tz_convert(eastern)
            new_data = {"Time": timestamp, "Price": live["last_price"]}
            live_df = pd.concat([live_df, pd.DataFrame([new_data])], ignore_index=True)
        time.sleep(1)

    if len(live_df) < 10:
        st.warning("Not enough data collected. Try again.")
        return

    # Prepare data
    live_df["TimeNumeric"] = live_df["Time"].apply(lambda x: x.timestamp())
    X = live_df["TimeNumeric"].values.reshape(-1, 1)
    y = live_df["Price"].values

    # Fit regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Forecast further into the future for more signal
    last_time = live_df["TimeNumeric"].iloc[-1]
    future_times = np.array([
        last_time + i for i in range(1, forecast_duration + 11)
    ]).reshape(-1, 1)  # Extra 10s forecast
    future_pred = model.predict(future_times)
    future_dates = [pd.to_datetime(t, unit='s').tz_localize("UTC").tz_convert(eastern)
                    for t in future_times.flatten()]

    # Confidence bounds
    residuals = y - y_pred
    std_dev = np.std(residuals)
    upper_bound = future_pred + std_dev
    lower_bound = future_pred - std_dev

    # Movement prediction from later point (forecast_duration + 10)
    final_pred = future_pred[-1]
    current_price = y[-1]
    delta = final_pred - current_price
    movement = "Up" if delta > std_dev else "Down" if delta < -std_dev else "Hold"

    # Plotting
    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(x=live_df["Time"], y=y, mode="markers", name="Actual Price"))
    fig_lr.add_trace(go.Scatter(x=live_df["Time"], y=y_pred, mode="lines", name="Fitted Line"))
    fig_lr.add_trace(go.Scatter(x=future_dates, y=future_pred, mode="lines", line=dict(dash="dash"), name="Forecast"))
    fig_lr.add_trace(go.Scatter(x=future_dates, y=upper_bound, mode="lines", line=dict(dash="dot"), name="Upper Bound"))
    fig_lr.add_trace(go.Scatter(x=future_dates, y=lower_bound, mode="lines", line=dict(dash="dot"), name="Lower Bound"))

    fig_lr.update_layout(
        title=f"{forecast_duration + 10}-Second Forecast with Confidence Bounds",
        xaxis_title="Time",
        yaxis_title="Price (USD)"
    )
    fig_lr.update_xaxes(tickformat="%H:%M:%S")

    st.plotly_chart(fig_lr, use_container_width=True)

    # Display summary
    st.success(f"Predicted movement: **{movement}**")
    st.write(f"Last price: ${current_price:.2f}")
    st.write(f"Forecasted price in {forecast_duration + 10}s: ${final_pred:.2f} ± {std_dev:.2f}")
