import streamlit as st
import pandas as pd
import yfinance as yf

def run_scanner():
    st.header("🔍 Swing Scanner")

    symbols = st.text_area("Enter stock symbols (comma-separated)", value="AAPL, TSLA, MSFT, NVDA").split(",")
    interval = st.selectbox("Select Timeframe", ["1d", "1h", "15m"])
    min_rsi = st.slider("Minimum RSI", 0, 100, 30)
    max_rsi = st.slider("Maximum RSI", 0, 100, 70)

    scan_results = []

    for symbol in [s.strip().upper() for s in symbols if s.strip()]:
        try:
            df = yf.download(symbol, period="7d", interval=interval)
            if df.empty:
                continue
            df['rsi'] = compute_rsi(df['Close'])
            latest_rsi = df['rsi'].iloc[-1]

            if min_rsi <= latest_rsi <= max_rsi:
                scan_results.append({
                    "Symbol": symbol,
                    "RSI": round(latest_rsi, 2),
                    "Last Price": round(df['Close'].iloc[-1], 2)
                })

        except Exception as e:
            st.warning(f"Could not fetch data for {symbol}: {e}")

    if scan_results:
        st.success(f"✅ Found {len(scan_results)} matching symbols.")
        st.dataframe(pd.DataFrame(scan_results))
    else:
        st.warning("No stocks matched your criteria.")

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)
