import pandas as pd
import streamlit as st
import ccxt
import pandas_ta as ta
import yfinance as yf

exchange = ccxt.coinbase()

FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_14', 'EMA_14', 'RSI_14',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Middle', 'BB_Lower'
]

@st.cache_data
#def fetch_historical_data(ticker: str, timeframe: str = '1d', limit: int = 150) -> pd.DataFrame:
def fetch_historical_data(ticker: str, period: str = '1mo') -> pd.DataFrame:
    try:
        #ohlcv = exchange.fetch_ohlcv(ticker, timeframe, limit=limit)
        df = yf.download(ticker, period=period, multi_level_index=False)
        #df = pd.DataFrame(ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        #df['Date'] = pd.to_datetime(df[df.index], unit='ms')
        #df = df.sort_values(df['Time']).reset_index(drop=True)
        #df.fillna(method="ffill", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def fetch_live_price(ticker: str):
    try:
        ticker_data = exchange.fetch_ticker(ticker)
        return {
            "last_price": ticker_data["last"],
            "timestamp": pd.to_datetime(ticker_data["timestamp"], unit="ms")
        }
    except Exception as e:
        st.error(f"Error fetching live price: {e}")
        return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        st.warning("Not enough data to compute indicators.")
        return df

    df['SMA_14'] = ta.sma(df['Close'], length=14).fillna(method="bfill")
    df['EMA_14'] = ta.ema(df['Close'], length=14).fillna(method="bfill")
    df['RSI_14'] = ta.rsi(df['Close'], length=14).fillna(method="bfill")

    macd = ta.macd(df['Close'])
    if macd is not None:
        df['MACD'] = macd['MACD_12_26_9'].fillna(method="bfill")
        df['MACD_Signal'] = macd['MACDs_12_26_9'].fillna(method="bfill")
        df['MACD_Hist'] = macd['MACDh_12_26_9'].fillna(0)
    else:
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = 0, 0, 0

    bb = ta.bbands(df['Close'], length=20)
    if bb is not None:
        df['BB_Upper'] = bb['BBU_20_2.0'].fillna(method="bfill")
        df['BB_Middle'] = bb['BBM_20_2.0'].fillna(method="bfill")
        df['BB_Lower'] = bb['BBL_20_2.0'].fillna(method="bfill")
    else:
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = 0, 0, 0

    return df
