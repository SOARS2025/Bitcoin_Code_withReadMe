import kaleido
import plotly
import google.generativeai as genai
import pandas
import yfinance as yf
print(plotly.__version__)

df = yf.download("AAPL", period='1mo', multi_level_index=False)
print(df)