
# Swing Scanner module
import streamlit as st
import pandas as pd
import yfinance as yf
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor
from modules.entry_logic import calculate_trigger_and_flags, log_entry_analysis

@st.cache_data
def get_all_us_tickers():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)
    return df['Symbol'].tolist()

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
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    return df

@st.cache_resource
def train_model():
    tickers = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA']
    data = []
    for t in tickers:
        df = get_price_data(t)
        df = calculate_indicators(df)
        df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
        data.append(df[['RSI', 'MACD', 'Signal', 'Target']])
    all_data = pd.concat(data).dropna()
    X = all_data[['RSI', 'MACD', 'Signal']]
    y = all_data['Target']
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

def scan_stock(ticker, model, filters):
    try:
        df = get_price_data(ticker)
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        price = latest['Close']
        if price < 20 or price > 70 or latest['Volume'] < filters['volume_min']:
            return None
        if not (filters['rsi_min'] <= latest['RSI'] <= filters['rsi_max']):
            return None
        if filters['macd_filter'] and latest['MACD'] <= latest['Signal']:
            return None
        if abs(price - latest['EMA20']) / latest['EMA20'] * 100 > filters['ema_tolerance']:
            return None

        features = pd.DataFrame([[latest['RSI'], latest['MACD'], latest['Signal']]], columns=['RSI', 'MACD', 'Signal'])
        prob = model.predict_proba(features)[0][1]

        avg_volume = df['Volume'][-20:].mean()
        trigger_price, entry_valid, reasons = calculate_trigger_and_flags(latest, df, avg_volume)
        log_entry_analysis(ticker, trigger_price, entry_valid, reasons, prob * 100)

        return {
            "Ticker": ticker,
            "Price": round(price, 2),
            "RSI": round(latest['RSI'], 2),
            "Confidence": round(prob * 100, 2),
            "Entry Trigger": trigger_price,
            "Entry Valid": entry_valid,
            "Entry Reasons": reasons
        }
    except:
        return None

def run_scanner():
    st.subheader("Scan for Swing Trade Setups")
    rsi_min = st.slider("RSI Min", 0, 100, 40)
    rsi_max = st.slider("RSI Max", 0, 100, 60)
    macd_filter = st.checkbox("Require MACD > Signal", value=True)
    ema_tolerance = st.slider("Max Distance from EMA20 (%)", 0, 10, 3)
    volume_min = st.number_input("Minimum Volume", value=1_000_000, step=100_000)

    if st.button("Run Scan"):
        model = train_model()
        filters = {
            'rsi_min': rsi_min,
            'rsi_max': rsi_max,
            'macd_filter': macd_filter,
            'ema_tolerance': ema_tolerance,
            'volume_min': volume_min
        }
        tickers = get_all_us_tickers()
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda t: scan_stock(t, model, filters), tickers))
        results = [r for r in results if r]
        df = pd.DataFrame(results)
        if not df.empty:
            st.success(f"Found {len(df)} setups")
            st.dataframe(df)
        else:
            st.warning("No setups found.")
