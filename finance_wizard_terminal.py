import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objs as go
from datetime import datetime
import requests

st.set_page_config(page_title="Finance Wizard Swing Terminal", layout="wide")
st.title("📈 Finance Wizard Swing Trade Terminal (Optimized)")

@st.cache_data
def get_all_us_tickers():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)
    tickers = df['Symbol'].tolist()
    return tickers

@st.cache_data
def get_price_data(ticker):
    return yf.Ticker(ticker).history(period="6mo")

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_indicators(df):
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Momentum'] = df['Close'].diff()
    return df

@st.cache_resource
def train_global_model():
    ticker_list = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA']
    frames = []
    for ticker in ticker_list:
        df = get_price_data(ticker)
        df = calculate_indicators(df)
        df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
        frames.append(df[['RSI', 'MACD', 'Signal', 'Momentum', 'Target']])
    full_data = pd.concat(frames).dropna()
    X = full_data[['RSI', 'MACD', 'Signal', 'Momentum']]
    y = full_data['Target']
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

def scan_ticker(ticker, model, filters):
    try:
        df = get_price_data(ticker)
        if df.empty or len(df) < 30:
            return None
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        features = pd.DataFrame([[latest['RSI'], latest['MACD'], latest['Signal'], latest['Momentum']]],
                                columns=['RSI', 'MACD', 'Signal', 'Momentum'])
        pred = model.predict(features)[0]
        prob = round(model.predict_proba(features)[0][1] * 100, 2)

        price = latest['Close']
        ema = latest['EMA20']
        rsi = latest['RSI']
        macd = latest['MACD']
        signal = latest['Signal']
        volume = latest['Volume']
        near_ema = abs(price - ema) / ema * 100 <= filters['ema_tolerance']

        if price < 5 or price > 200 or volume < filters['volume_min']:
            return None

        if filters['rsi_min'] <= rsi <= filters['rsi_max'] and (not filters['macd_filter'] or macd > signal) and near_ema and prob >= filters['confidence_min']:
            status = "Swing ✅" if pred == 1 else "Approaching 🔄"
            return {"Ticker": ticker, "Price": round(price, 2), "RSI": round(rsi, 1), "AI Confidence %": prob, "Status": status}
    except:
        return None

# === UI ===
tab1, tab2 = st.tabs(["📈 Swing", "📓 Journal"])

with tab1:
    st.subheader("Optimized AI Swing Screener")

    rsi_min = st.slider("RSI Min", 10, 90, 40)
    rsi_max = st.slider("RSI Max", 10, 90, 60)
    macd_filter = st.checkbox("Require MACD > Signal", value=True)
    ema_tolerance = st.slider("Max Distance from EMA20 (%)", 0, 10, 3)
    volume_min = st.number_input("Min Volume", 100000, 10000000, 1000000, step=100000)
    confidence_min = st.slider("Minimum AI Confidence %", 50, 100, 60)
    run_scan = st.button("🔍 Run Optimized Scan")

    if run_scan:
        tickers = get_all_us_tickers()
        filters = {
            'rsi_min': rsi_min,
            'rsi_max': rsi_max,
            'macd_filter': macd_filter,
            'ema_tolerance': ema_tolerance,
            'volume_min': volume_min,
            'confidence_min': confidence_min
        }
        model = train_global_model()

        st.info("Scanning tickers with multithreading...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda t: scan_ticker(t, model, filters), tickers))

        results = [res for res in results if res is not None]
        df = pd.DataFrame(results)
        if not df.empty:
            st.success(f"Found {len(df)} setups")
            st.dataframe(df)
        else:
            st.warning("No setups found.")

with tab2:
    st.subheader("Trade Journal")
    journal_file = "trade_log.csv"
    try:
        journal_df = pd.read_csv(journal_file)
    except:
        journal_df = pd.DataFrame(columns=["Ticker", "Entry Date", "Entry Price", "Exit Date", "Exit Price", "Return %", "Days Held", "Notes"])
    with st.form("trade_entry_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input("Ticker").upper()
            entry_price = st.number_input("Entry Price", min_value=0.01)
            entry_date = st.date_input("Entry Date", value=datetime.today())
        with col2:
            exit_price = st.number_input("Exit Price", min_value=0.01)
            exit_date = st.date_input("Exit Date", value=datetime.today())
        with col3:
            notes = st.text_area("Notes")
        submitted = st.form_submit_button("Add Trade")
        if submitted and ticker and entry_price and exit_price:
            return_pct = round(((exit_price - entry_price) / entry_price) * 100, 2)
            days_held = (exit_date - entry_date).days
            new_row = pd.DataFrame([[ticker, entry_date, entry_price, exit_date, exit_price, return_pct, days_held, notes]],
                                   columns=journal_df.columns)
            journal_df = pd.concat([journal_df, new_row], ignore_index=True)
            journal_df.to_csv(journal_file, index=False)
            st.success(f"Trade for {ticker} added.")
    st.dataframe(journal_df)
    st.download_button("Download Journal", journal_df.to_csv(index=False), file_name="trade_log.csv")
