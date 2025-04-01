import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go
from datetime import datetime
import requests

st.set_page_config(page_title="Finance Wizard Swing Terminal", layout="wide")
st.title("📈 Finance Wizard Swing Trade Terminal")

@st.cache_data
def get_all_us_tickers():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)
    tickers = df['Symbol'].tolist()
    return tickers

# === Helper Functions ===
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

def train_model(data):
    data = data.dropna()
    data['Target'] = (data['Close'].shift(-3) > data['Close']).astype(int)
    features = ['RSI', 'MACD', 'Signal', 'Momentum']
    X = data[features]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def plot_chart(ticker):
    df = yf.Ticker(ticker).history(period="3mo")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=20).mean(), mode='lines', name='EMA20'))
    fig.update_layout(title=f'{ticker} Chart', xaxis_title='Date', yaxis_title='Price', height=400)
    return fig

# === Tabs ===
tab1, tab2 = st.tabs(["📈 Swing", "📓 Journal"])

with tab1:
    st.subheader("Swing Trade Screener with Smart AI Forecast")

    default_rsi = (40, 60)
    default_ema_tolerance = 3
    default_confidence = 60
    rsi_min, rsi_max = st.slider("RSI Range", 10, 90, default_rsi)
    macd_filter = st.checkbox("Require MACD > Signal", value=True)
    ema_tolerance = st.slider("Max Distance from EMA20 (%)", 0, 10, default_ema_tolerance)
    volume_min = st.number_input("Min Volume", 100000, 10000000, 1000000, step=100000)
    confidence_min = st.slider("Minimum AI Confidence %", 50, 100, default_confidence)
    if st.button("Reset to Defaults"):
        rsi_min, rsi_max = default_rsi
        ema_tolerance = default_ema_tolerance
        confidence_min = default_confidence

    run_scan = st.button("🔍 Run Scan")
    if run_scan:
        tickers = get_all_us_tickers()
        confirmed, approaching = [], []

        for ticker in tickers:
            try:
                df = yf.Ticker(ticker).history(period="6mo")
                if df.empty or len(df) < 30:
                    continue
                df = calculate_indicators(df)
                model = train_model(df)

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
                near_ema = abs(price - ema) / ema * 100 <= ema_tolerance

                if price < 5 or price > 200 or volume < volume_min:
                    continue

                if rsi_min <= rsi <= rsi_max and (not macd_filter or macd > signal) and near_ema and prob >= confidence_min:
                    if pred == 1:
                        confirmed.append({"Ticker": ticker, "Price": round(price, 2), "RSI": round(rsi, 1), "AI Confidence %": prob, "Status": "Swing ✅"})
                    else:
                        approaching.append({"Ticker": ticker, "Price": round(price, 2), "RSI": round(rsi, 1), "AI Confidence %": prob, "Status": "Approaching 🔄"})
            except:
                continue

        if confirmed:
            st.success("✅ Confirmed Swing Setups")
            st.dataframe(pd.DataFrame(confirmed))

        if approaching:
            st.info("🔮 AI Forecasted Approaching Setups")
            st.dataframe(pd.DataFrame(approaching))

        if not confirmed and not approaching:
            st.warning("No setups found with current filters.")

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
