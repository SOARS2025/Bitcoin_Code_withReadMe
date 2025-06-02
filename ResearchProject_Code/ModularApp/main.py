# Streamlit and standard modules
import streamlit as st
import os
import time
from typing import Tuple, Any
import re

# Data handling
import pandas as pd
import numpy as np

# Visualization
import plotly.graph_objs as go

# Technical Analysis
import pandas_ta as ta

# Web scraping for sentiment
import requests
from bs4 import BeautifulSoup

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# Crypto data
import ccxt

# NLP (Sentiment)
from google.cloud import language_v1

# --------------------------
# Constants & Global Variables
# --------------------------
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_14', 'EMA_14', 'RSI_14',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Middle', 'BB_Lower'
]

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = st.secrets["GOOGLE_CLOUD_CREDENTIALS"]

# --------------------------
# Initialize Exchange & Sidebar
# --------------------------
exchange = ccxt.coinbase()
tab = st.sidebar.radio(
    "Select View", 
    ["Live Price", "Market Analysis", "Live Regression", "Trained Model", "Sentiment Analysis"],
    key="main_tab_selector"
)


# --------------------------
# Import External Modules for Tabs
# --------------------------
from trained_model_tab import run_trained_model_tab
from live_price_tab import run_live_price_tab
from market_analysis_tab import run_market_analysis_tab
from live_regression_tab import run_live_regression_tab
from sentiment_analysis_tab import run_sentiment_analysis_tab



# --------------------------
# Run Tab Content
# --------------------------
if tab == "Trained Model":
    run_trained_model_tab()
elif tab == "Live Price":
    run_live_price_tab()
elif tab == "Market Analysis":
    run_market_analysis_tab()
elif tab == "Live Regression":
    run_live_regression_tab()
elif tab == "Sentiment Analysis":
    run_sentiment_analysis_tab()

