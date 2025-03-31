import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Finance Wizard Terminal", layout="wide")
st.title("🧙‍♂️ Finance Wizard Terminal")

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
tab1, tab2, tab3, tab4 = st.tabs(["🧠 Options", "📈 Swing", "📓 Journal", "🔁 Backtest"])

with tab1:
    st.subheader("Options Spread Scanner (Demo Placeholder)")
    st.info("Coming soon: Your options scanner with ROI filters and AI overlay.")

with tab2:
    st.subheader("Swing Trade Screener")
    price_min = st.slider("Min Price", 5, 100, 20)
    price_max = st.slider("Max Price", 20, 200, 70)
    min_volume = st.number_input("Min Volume (Last Day)", 100000, 5000000, 1000000, step=100000)
    simulated_mode = st.checkbox("Use Previous Day's Data", value=False)
    candidates = ["AAPL", "AMD", "NVDA", "TSLA", "MSFT", "GOOGL", "META"]
    results = []
    for ticker in candidates:
        data = yf.Ticker(ticker).history(period="6mo")
        if data.empty:
            continue
        data = calculate_indicators(data)
        latest = data.iloc[-2] if simulated_mode else data.iloc[-1]
        if latest['Close'] < price_min or latest['Close'] > price_max:
            continue
        if latest['Volume'] < min_volume:
            continue
        if latest['Close'] > latest['EMA20'] and 40 < latest['RSI'] < 60 and latest['MACD'] > latest['Signal']:
            trend, confidence = predict_trend(ticker)
            if trend == 'Up':
                results.append({"Ticker": ticker, "Price": round(latest['Close'], 2), "RSI": round(latest['RSI'], 1), "Confidence": confidence, "Signal": "Swing ✅"})
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
        ticker_choice = st.selectbox("View Chart:", df['Ticker'])
        st.plotly_chart(plot_chart(ticker_choice))
    else:
        st.warning("No swing setups matched.")

with tab3:
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

with tab4:
    st.subheader("Backtest (Simulated)")
    backtest_days = 90
    holding_period = 5
    entry_price = 100
    target_pct = st.selectbox("Target Gain %", [3, 5, 7, 10], index=1)
    stop_loss = st.checkbox("Include -3% Stop-Loss", value=True)
    data = []
    for day in range(backtest_days):
        entry_day = datetime.today() - timedelta(days=day + holding_period)
        fake_price = entry_price * (1 + (0.01 * (day % 6 - 3)))
        hit_target = fake_price >= entry_price * (1 + target_pct / 100)
        stopped_out = fake_price <= entry_price * 0.97
        result = "Target Hit ✅" if hit_target else ("Stopped ❌" if stop_loss and stopped_out else "Held")
        pct_return = round(((fake_price - entry_price) / entry_price) * 100, 2)
        data.append({"Entry Date": entry_day.date(), "Entry Price": entry_price, "Exit Price": round(fake_price, 2), "% Return": pct_return, "Outcome": result})
    backtest_df = pd.DataFrame(data)
    st.dataframe(backtest_df)
    st.download_button("Download Backtest", backtest_df.to_csv(index=False), file_name="backtest_results.csv")
