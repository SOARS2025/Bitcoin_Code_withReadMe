import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import pandas_ta as ta
import ccxt

from app_utils import fetch_live_price, fetch_historical_data, add_indicators
#replacing all data['Time'] with data.index

exchange = ccxt.coinbase()

def plot_volume_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Volume", title="Trading Volume")
    return fig

def plot_obv_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], mode="lines", name="OBV"))
    fig.update_layout(xaxis_title="Date", yaxis_title="OBV", title="On-Balance Volume (OBV)")
    return fig

def plot_candlestick_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name="Candlestick")])
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='rgba(255,0,0,0.5)'), name="BB Upper"))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], line=dict(color='rgba(0,0,255,0.5)'), name="BB Middle"))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='rgba(0,255,0,0.5)'), name="BB Lower"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", title="Candlestick Chart with Bollinger Bands")
    return fig

def plot_ma_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Close Price"))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_14'], mode='lines', name="SMA 14"))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_14'], mode='lines', name="EMA 14"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", title="Price with SMA & EMA")
    return fig

def plot_rsi_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI_14'], mode='lines', name='RSI'))
    fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text="Overbought")
    fig.add_hline(y=30, line_dash='dash', line_color='green', annotation_text="Oversold")
    fig.update_layout(xaxis_title="Date", yaxis_title="RSI", title="Relative Strength Index (RSI)",
                      yaxis=dict(range=[data['RSI_14'].min()-5, data['RSI_14'].max()+5]))
    return fig

def plot_macd_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD Line'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Signal Line'))
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='MACD Histogram', opacity=0.5))
    fig.update_layout(xaxis_title="Date", yaxis_title="MACD", title="MACD Indicator")
    return fig

def run_market_analysis_tab():
    st.subheader("Market Analysis")
    #selected_asset = st.sidebar.selectbox("Choose a Coin:", ["Bitcoin (BTC)"])
    selected_asset = st.sidebar.selectbox("Choose a Stock:", ["Apple (AAPL)"])
    #ticker = "BTC/USD"
    ticker = "AAPL"
    #data = fetch_historical_data(ticker, timeframe='1d', limit=150)
    data = fetch_historical_data(ticker, period='1y')
    if data.empty:
        st.write("No historical data available.")
        return

    data = add_indicators(data)
    data['OBV'] = ta.obv(data['Close'], data['Volume'])

    st.markdown("### Market Summary")
    latest = data.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("Open", f"{latest['Open']:.2f}")
    cols[1].metric("High", f"{latest['High']:.2f}")
    cols[2].metric("Low", f"{latest['Low']:.2f}")
    cols[3].metric("Close", f"{latest['Close']:.2f}")

    st.markdown("### Additional Indicators")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Volume")
        st.plotly_chart(plot_volume_chart(data), use_container_width=True)
    with col2:
        st.markdown("#### On-Balance Volume (OBV)")
        st.plotly_chart(plot_obv_chart(data), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Candlestick Chart with Bollinger Bands")
        st.plotly_chart(plot_candlestick_chart(data), use_container_width=True)
    with col4:
        st.markdown("### SMA & EMA Overlay on Price Chart")
        st.plotly_chart(plot_ma_chart(data), use_container_width=True)

    st.markdown("### Relative Strength Index (RSI)")
    st.plotly_chart(plot_rsi_chart(data), use_container_width=True)

    st.markdown("### MACD Indicator")
    st.plotly_chart(plot_macd_chart(data), use_container_width=True)