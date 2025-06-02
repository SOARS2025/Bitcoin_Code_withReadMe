Capstone Research Project: A Systems of Algorithms to Assess Crypto Market Predictability

Abstract:
	The goal and research question of this project is to answer the question: Can cryptocurrency markets be accurately predicted using a system of individual algorithms? To answer this question, I designed and built an application that shows how multiple models can be merged and used together to analyze the crypto market movements as a whole
	The application maintains an educational and interactive approach, allowing the user to understand each algorithm and the insights it provides. This allows the user to understand the predictions and gain trust in the results, showing users the full process, from data collection and feature engineering to model training and live forecasting. 
	It includes tools for feature selection, a random forest algorithm, a linear regression model, and a sentiment analysis module powered by a generative language model. 

The app features:
- Live BTC/USD price feed visualization
- Live short-term regression forecasting
- Trained Random Forest model on historical data
- Sentiment analysis using Google Cloud Natural Language API
- Technical indicators and market analysis tools



Architecture Overview

main.py
 ├── live_price_tab.py        # Displays live BTC/USD price feed
 ├── live_regression_tab.py    # Collects live prices, fits regression model, forecasts trend
 ├── market_analysis_tab.py    # Technical analysis (RSI, MACD, OBV, Bollinger Bands)
 ├── trained_model_tab.py      # Trains and evaluates a Random Forest model
 └── sentiment_analysis_tab.py # Sentiment analysis from preset crypto news sites

Utilities:
- app_utils.py           # Functions for data fetching and technical indicators
- Gemini_BTCUSD_d.csv    # Historical BTC/USD price data for model training
- trained_model_results.csv # Results from Random Forest model evaluations
- requirements.txt       # List of Python packages



Install the required packages, use following command with appropriate path:

pip install -r requirements.txt

Packages used:
- streamlit
- pandas
- numpy
- scikit-learn
- pandas-ta
- plotly
- ccxt
- beautifulsoup4
- google-cloud-language
- requests




How to Run

1. Install all required packages:

2. Set up the `secrets.toml` file:
   - Navigate to `.streamlit/secrets.toml`
   - Add your Google Cloud credentials path:

[default]
GOOGLE_CLOUD_CREDENTIALS = "path/to/your/credentials.json"


3. Run the application:

streamlit run ModularApp\main.py


4. Use the sidebar to navigate between Live Price, Market Analysis, Live Regression, Trained Model, and Sentiment Analysis tabs.




Known Bugs and Problems

Sentiment analysis sometimes will get rate-limited from websites it scrapes
Live-price can be inaccurate and output price every few seconds and not every second
Random Forest is largely over-fitted needs to be tweaked for it to properly train on data. Structure is there


Contact
Marco Azzani  
marco.azzani@outlook.com

