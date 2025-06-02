import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

df = pd.read_parquet("data.parquet")
df.set_index("datetime", inplace=True)
df.sample(10) # print 10 random rows from the dataset