import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import time
import pytz
from typing import Any
import yfinance as yf

from app_utils import fetch_live_price, fetch_historical_data

def run_live_price_tab():
    st.subheader("Live Price")
    selected_asset = st.sidebar.selectbox("Choose a Coin:", ["Bitcoin (BTC)"])
    ticker = "BTC/USD"

    st.write("Live BTC/USD price updates are shown below for 60 seconds to verify real-time data feed.")

    eastern = pytz.timezone("America/New_York")
    live_data = pd.DataFrame(columns=["Time", "Price"])

    price_placeholder = st.empty()
    chart_placeholder = st.empty()

    previous_price = None
    start_time = time.time()

    while time.time() - start_time < 60:
        live_price = fetch_live_price(ticker)
        if live_price:
            # Handle timestamp correctly
            ts = pd.to_datetime(live_price["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts = ts.tz_convert(eastern)

            price = live_price["last_price"]
            new_row = pd.DataFrame([[ts, price]], columns=["Time", "Price"])
            live_data = pd.concat([live_data, new_row], ignore_index=True).tail(60)

            # Compute Δ and trend
            delta = price - previous_price if previous_price is not None else 0
            trend_symbol = "⬆️" if delta > 0 else "⬇️" if delta < 0 else "➡️"
            previous_price = price

            # Show price info
            formatted_time = ts.strftime("%I:%M:%S %p")
            price_placeholder.markdown(
                f"**Current Price:** ${price:.2f} {trend_symbol} (Δ {delta:+.2f}) at {formatted_time} EDT"
            )

            # Rolling chart
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(
                x=live_data["Time"],
                y=live_data["Price"],
                mode="lines+markers",
                name="Live Price"
            ))
            fig_live.update_layout(
                title="BTC/USD Live Price Feed (Rolling 60 Seconds)",
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                margin=dict(l=30, r=30, t=30, b=30),
                xaxis=dict(
                    tickformat="%I:%M:%S %p",
                    range=[live_data["Time"].min(), live_data["Time"].max()]
                )
            )
            chart_placeholder.plotly_chart(fig_live, use_container_width=True)

        time.sleep(1)

    # Historical OHLC chart
    st.markdown("### Historical OHLC Data")
    hist_data = fetch_historical_data(ticker, timeframe='1d', limit=150)
    if not hist_data.empty:
        fig_ohlc = go.Figure(data=[go.Candlestick(
            x=hist_data.index, #x=hist_data['Time']
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name="OHLC"
        )])
        fig_ohlc.update_layout(
            title="Historical OHLC Data",
            xaxis_title="Time",
            yaxis_title="Price (USD)"
        )
        st.plotly_chart(fig_ohlc, use_container_width=True)
