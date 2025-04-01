import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime
import requests
import io

st.set_page_config(page_title="Finance Wizard Swing Terminal", layout="wide")
st.title("📈 Finance Wizard Swing Trade Terminal")

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
    return df

def predict_trend(ticker):
    try:
        data = yf.Ticker(ticker).history(period="6mo")
        data = data.dropna()
        data['days'] = np.arange(len(data))
        model = LinearRegression()
        model.fit(data[['days']], data['Close'])
        next_day = len(data)
        prediction = model.predict(pd.DataFrame([[next_day]], columns=['days']))[0]
        current = data['Close'].iloc[-1]
        confidence = abs(prediction - current) / current * 100
        return 'Up' if prediction > current else 'Down', round(confidence, 2)
    except:
        return 'N/A', 0

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

@st.cache_data
def get_all_us_tickers():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents_symbols.txt"
    df = pd.read_csv(url)
    tickers = df['Symbol'].tolist()
    return tickers


# === Tabs ===
tab1, tab2 = st.tabs(["📈 Swing", "📓 Journal"])

with tab1:
    st.subheader("Swing Trade Screener with AI Forecast")
    price_min = st.slider("Min Price", 5, 100, 20)
    price_max = st.slider("Max Price", 20, 200, 70)
    min_volume = st.number_input("Min Volume (Last Day)", 100000, 5000000, 1000000, step=100000)
    simulated_mode = st.checkbox("Use Previous Day's Data", value=False)
    run_scan = st.button("🔍 Run Scan")

    if run_scan:
        tickers = get_all_us_tickers()
        confirmed = []
        approaching = []

        for ticker in tickers:
            try:
                data = yf.Ticker(ticker).history(period="6mo")
                if data.empty or len(data) < 30:
                    continue
                data = calculate_indicators(data)
                latest = data.iloc[-2] if simulated_mode else data.iloc[-1]

                price = latest['Close']
                rsi = latest['RSI']
                macd = latest['MACD']
                signal = latest['Signal']
                ema = latest['EMA20']
                volume = latest['Volume']
                trend, confidence = predict_trend(ticker)

                if price < price_min or price > price_max or volume < min_volume:
                    continue

                if price > ema and 40 < rsi < 60 and macd > signal and trend == 'Up':
                    confirmed.append({"Ticker": ticker, "Price": round(price, 2), "RSI": round(rsi, 1), "Trend": trend, "Confidence": confidence, "Status": "Swing ✅"})
                elif 35 < rsi < 65 and abs(price - ema) / ema < 0.03:
                    approaching.append({"Ticker": ticker, "Price": round(price, 2), "RSI": round(rsi, 1), "Trend": trend, "Confidence": confidence, "Status": "Approaching Setup 🔄"})
            except:
                continue

        if confirmed:
            st.success("🎯 Confirmed Swing Trade Setups")
            df1 = pd.DataFrame(confirmed)
            st.dataframe(df1)

        if approaching:
            st.info("🔮 AI Forecast: Approaching Setups")
            df2 = pd.DataFrame(approaching)
            st.dataframe(df2)

        if not confirmed and not approaching:
            st.warning("No setups found based on current filters.")

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
