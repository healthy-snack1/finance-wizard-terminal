# Trade Journal module
import streamlit as st
import pandas as pd
import os
from datetime import datetime

def run_journal():
    st.subheader("📓 Trade Journal")
    journal_file = "data/trade_log.csv"
    if os.path.exists(journal_file):
        journal_df = pd.read_csv(journal_file)
    else:
        journal_df = pd.DataFrame(columns=["Ticker", "Entry Date", "Entry Price", "Shares", "Exit Date", "Exit Price", "Profit $", "Profit %", "Hold Days", "Status", "Notes"])

    st.dataframe(journal_df)
    with st.expander("➕ Add Trade"):
        ticker = st.text_input("Ticker")
        entry_date = st.date_input("Entry Date", datetime.today())
        entry_price = st.number_input("Entry Price", min_value=0.01)
        shares = st.number_input("Shares", min_value=1, value=10)
        exit_price = st.number_input("Exit Price", value=0.0)
        exit_date = st.date_input("Exit Date", datetime.today())
        notes = st.text_area("Notes")
        if st.button("Add to Journal"):
            profit_amt = (exit_price - entry_price) * shares if exit_price > 0 else 0
            profit_pct = ((exit_price - entry_price) / entry_price) * 100 if exit_price > 0 else 0
            hold_days = (exit_date - entry_date).days if exit_price > 0 else 0
            status = "Closed" if exit_price > 0 else "Open"
            new_row = pd.DataFrame([[ticker, entry_date, entry_price, shares, exit_date, exit_price, round(profit_amt, 2), round(profit_pct, 2), hold_days, status, notes]],
                                   columns=journal_df.columns)
            journal_df = pd.concat([journal_df, new_row], ignore_index=True)
            journal_df.to_csv(journal_file, index=False)
            st.success("Trade added.")
