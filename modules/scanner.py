
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor
from modules.entry_trigger import get_entry_trigger

@st.cache_data
def get_all_us_tickers():
    try:
        nasdaq_url = "https://old.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        other_url = "https://old.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        nasdaq = pd.read_csv(nasdaq_url, sep="|")
        other = pd.read_csv(other_url, sep="|")
        tickers = pd.concat([nasdaq["Symbol"], other["ACT Symbol"]]).dropna().unique().tolist()
        return [t for t in tickers if t.isalpha()]
    except:
        return ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD']
def calculate_indicators(df):
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['VolumeAvg20'] = df['Volume'].rolling(20).mean()
    return df

def train_model():
    tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'ROKU']
    data = []
    for t in tickers:
        df = get_price_data(t)
        df = calculate_indicators(df)
        df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
        data.append(df[['Close', 'EMA20', 'EMA50', 'Volume', 'VolumeAvg20', 'Target']])
    all_data = pd.concat(data).dropna()
    X = all_data[['Close', 'EMA20', 'EMA50', 'Volume', 'VolumeAvg20']]
    y = all_data['Target']
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

def analyze_stock(ticker, model):
    try:
        df = get_price_data(ticker)
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        price = latest['Close']
        ema20 = latest['EMA20']
        ema50 = latest['EMA50']

        # Trend condition
        if not (price > ema20 > ema50):
            return None

        # Pullback condition
        recent_high = df['Close'].rolling(5).max().iloc[-2]
        if price > recent_high:
            return None  # avoid extended names

        # Relative volume
        if latest['Volume'] < 1.3 * latest['VolumeAvg20']:
            return None

        features = pd.DataFrame([[
            price, ema20, ema50, latest['Volume'], latest['VolumeAvg20']
        ]], columns=['Close', 'EMA20', 'EMA50', 'Volume', 'VolumeAvg20'])

        prob = model.predict_proba(features)[0][1]
        trigger_info = get_entry_trigger(ticker)

        return {
            'Ticker': ticker,
            'Price': round(price, 2),
            'Trend': 'Uptrend',
            'Rel Volume': round(latest['Volume'] / latest['VolumeAvg20'], 2),
            'Confidence': round(prob * 100, 2),
            'Trigger': trigger_info
        }
    except:
        return None

def run_scanner():
    st.subheader("🚀 ARK-Style AI Swing Scanner")

    tickers = get_all_us_tickers()
    model = train_model()

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(analyze_stock, t, model): t for t in tickers}
        for future in future_to_ticker:
            result = future.result()
            if result:
                results.append(result)

    if results:
        results.sort(key=lambda x: x['Confidence'], reverse=True)
        for res in results:
            with st.expander(f"{res['Ticker']} | Confidence: {res['Confidence']}% | Trend: {res['Trend']}"):
                st.write(f"**Current Price:** ${res['Price']}")
                st.write(f"**Relative Volume:** {res['Rel Volume']}x")
                if res['Trigger']:
                    for k, v in res['Trigger'].items():
                        st.write(f"**{k}:** {v}")
    else:
        st.warning("No ARK-style setups found today.")
