import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objs as go
import random
import pandas_ta as ta
import os
from app_utils import fetch_historical_data #I added this
#Changed all the ['close'] to ['Close'] along with the open, high, low stuff 
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "trained_model_results.csv")

def run_trained_model_tab():
    #st.title("BTC/USD Close Price Prediction")
    st.title("AAPL Close Price Prediction")

    # file_path = os.path.join(os.path.dirname(__file__), "Gemini_BTCUSD_d.csv")

    # @st.cache_data
    # def load_data(path):
    #     df = pd.read_csv(path, skiprows=1)
    #     df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    #     df['date'] = pd.to_datetime(df['date']).dt.date
    #     df = df.sort_values('date').reset_index(drop=True)
    #     df = df[df['volume_usd'] > 0]
    #     return df

    #df = load_data(file_path)
    df = fetch_historical_data("AAPL", period='1y')

    st.subheader("Raw Cleaned Data")
    st.dataframe(df.tail(10))

    st.subheader("Feature Engineering")

    df['date'] = df.index #added
    series = pd.Series(df['Volume']) #added
    df['volume'] = series.astype('float64') #added
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['rolling_mean_5'] = df['Close'].rolling(window=5).mean()
    df['rolling_mean_10'] = df['Close'].rolling(window=10).mean()
    df['momentum'] = df['Close'] - df['Close'].shift(3)
    df['price_range'] = df['High'] - df['Low']
    df['body_size'] = abs(df['Close'] - df['Open'])

    indicator_window = 14
    df['sma_14'] = ta.sma(df['Close'], length=indicator_window)
    df['ema_14'] = ta.ema(df['Close'], length=indicator_window)
    df['rsi_14'] = ta.rsi(df['Close'], length=indicator_window)
    macd = ta.macd(df['Close'])
    if macd is not None:
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
    bb = ta.bbands(df['Close'], length=20)
    if bb is not None:
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_middle'] = bb['BBM_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']

    df['target'] = df['Close'].shift(-1)
    df = df.dropna().reset_index(drop=True)

    features = ['Open', 'High', 'Low', 'Close', 'volume', 
                'returns', 'volatility', 'rolling_mean_5', 
                'rolling_mean_10', 'momentum', 'price_range', 'body_size',
                'sma_14', 'ema_14', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower']

    X = df[features]
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("Training Random Forest Model")
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    model = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=tscv, scoring='r2')
    with st.spinner("Training Random Forest model... performing grid search and time-series cross-validation. Please wait..."):
        grid.fit(X_scaled, y)
        best_model = grid.best_estimator_

    st.success(f"Best Parameters: {grid.best_params_}")

    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
    st.markdown(f"**Cross-Validation R² Scores (5-Fold):** {np.round(cv_scores, 4)}")
    st.markdown(f"**Mean CV R²:** {cv_scores.mean():.4f} | **Std Dev:** {cv_scores.std():.4f}")

    st.subheader("Testing on 14 Random Past Dates")
    random_indices = sorted(random.sample(range(len(df) - 1), 14))
    test_samples = df.iloc[random_indices]
    X_test = scaler.transform(test_samples[features])
    y_true = df.iloc[random_indices]['target'].values
    y_pred = best_model.predict(X_test)

    percent_error = np.abs((y_true - y_pred) / y_true) * 100

    results = pd.DataFrame({
        'Date': df.iloc[random_indices]['date'].astype(str),
        'Actual Close ($)': np.round(y_true, 2),
        'Predicted Close ($)': np.round(y_pred, 2),
        'Error ($)': np.round(np.abs(y_true - y_pred), 2),
        'Percent Error (%)': np.round(percent_error, 2)
    })

    st.table(results)

    # Save results to CSV for future tab
    results.to_csv(RESULTS_FILE, index=False)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    st.markdown("### Evaluation Metrics on Sampled Dates")
    st.markdown(f"- **MSE:** {mse:.4f}")
    st.markdown(f"- **MAE:** {mae:.4f}")
    st.markdown(f"- **R²:** {r2:.4f}")

    st.subheader("Recent Data and Model Prediction")
    recent = df.iloc[-30:].copy()
    X_recent_scaled = scaler.transform(recent[features])
    y_recent_pred = best_model.predict(X_recent_scaled)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent['date'], y=recent['Close'], name='Actual Close ($)', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=recent['date'], y=y_recent_pred, name='Predicted Close ($)', mode='lines+markers'))
    fig.update_layout(title='Recent Close Price & Model Prediction', xaxis_title='Date', yaxis_title='Close Price ($)')
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Explanation of Optional Technical Indicators"):
        st.markdown("""
        - **SMA_14 (Simple Moving Average)**: Smooths price over 14 days. Helps capture trend direction.
        - **EMA_14 (Exponential Moving Average)**: Like SMA, but weights recent prices more. Reacts faster to changes.
        - **RSI_14 (Relative Strength Index)**: Oscillates between 0–100. High = overbought, low = oversold. Momentum tool.
        - **MACD / Signal / Histogram**: Combines EMAs to show momentum and potential reversals.
        - **Bollinger Bands (Upper/Middle/Lower)**: Shows volatility. Price near upper/lower bands can signal strong moves.
        """)
