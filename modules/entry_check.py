import streamlit as st
from modules.multi_timeframe import is_signal_confirmed
from logic.entry_logic import should_enter_trade
from utils.fetch_data import fetch_ohlcv_data
from utils.indicators import compute_rsi

def run_entry_check():
    st.header("📥 Entry Signal Checker")

    symbol = st.text_input("Enter stock symbol (e.g., AAPL)")
    use_multitf = st.checkbox("Use Multi-Timeframe Confirmation", value=True)

    if st.button("Check Entry Signal"):
        df = fetch_ohlcv_data(symbol, interval="5m")

        if df is None or df.empty:
            st.error("No data found for this symbol.")
        else:
            df['rsi'] = compute_rsi(df['close'])
            volume = df['volume'].iloc[-1]
            volume_avg = df['volume'].rolling(20).mean().iloc[-1]
            rsi = df['rsi'].iloc[-1]

            st.markdown(f"**Latest RSI (5m):** {rsi:.2f}")
            st.markdown(f"**Latest Volume:** {volume:.2f}")
            st.markdown(f"**20-Period Volume Avg:** {volume_avg:.2f}")

            entry_data = {
                "rsi": rsi,
                "volume": volume,
                "volume_avg": volume_avg
            }

            if should_enter_trade(symbol, entry_data, use_multitf=use_multitf):
                st.success(f"✅ Entry signal CONFIRMED for {symbol}")
            else:
                st.warning(f"❌ No valid entry for {symbol} at this time")
