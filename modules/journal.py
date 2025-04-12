
import streamlit as st
import pandas as pd
import os
from datetime import datetime

LOG_PATH = "data/trade_journal.csv"

def init_journal():
    columns = [
        "Date Entered", "Ticker", "Entry Price", "Shares",
        "Date Exited", "Exit Price", "Profit %", "Profit $", "Hold Days", "Status"
    ]
    if not os.path.exists(LOG_PATH):
        df = pd.DataFrame(columns=columns)
        df.to_csv(LOG_PATH, index=False)

def run_journal():
    st.subheader("📓 Trade Journal")

    init_journal()

    df = pd.read_csv(LOG_PATH)
    st.dataframe(df, use_container_width=True)

    st.markdown("### ➕ Log New Trade")
    with st.form("log_trade_form"):
        ticker = st.text_input("Ticker").upper()
        entry_price = st.number_input("Entry Price", step=0.01)
        shares = st.number_input("Number of Shares", step=1)
        date_entered = st.date_input("Date Entered")

        submitted = st.form_submit_button("Add Trade")
        if submitted and ticker:
            new_entry = {
                "Date Entered": date_entered,
                "Ticker": ticker,
                "Entry Price": entry_price,
                "Shares": shares,
                "Date Exited": "",
                "Exit Price": "",
                "Profit %": "",
                "Profit $": "",
                "Hold Days": "",
                "Status": "Open"
            }
            df = df.append(new_entry, ignore_index=True)
            df.to_csv(LOG_PATH, index=False)
            st.success(f"Trade for {ticker} added.")

    st.markdown("### 📝 Update Exits")
    open_trades = df[df["Status"] == "Open"]
    if not open_trades.empty:
        ticker_to_update = st.selectbox("Select Ticker to Exit", open_trades["Ticker"].unique())
        exit_price = st.number_input("Exit Price", step=0.01)
        date_exited = st.date_input("Date Exited")

        if st.button("Update Exit"):
            idx = df[df["Ticker"] == ticker_to_update].index[-1]
            df.at[idx, "Date Exited"] = date_exited
            df.at[idx, "Exit Price"] = exit_price
            entry_price = df.at[idx, "Entry Price"]
            shares = df.at[idx, "Shares"]
            df.at[idx, "Profit %"] = round((exit_price - entry_price) / entry_price * 100, 2)
            df.at[idx, "Profit $"] = round((exit_price - entry_price) * shares, 2)
            days_held = (pd.to_datetime(str(date_exited)) - pd.to_datetime(str(df.at[idx, "Date Entered"]))).days
            df.at[idx, "Hold Days"] = days_held
            df.at[idx, "Status"] = "Closed"
            df.to_csv(LOG_PATH, index=False)
            st.success(f"{ticker_to_update} exit updated.")
    else:
        st.info("No open trades to update.")
