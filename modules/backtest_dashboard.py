
import streamlit as st
import pandas as pd

def run_backtest_dashboard():
    st.subheader("📊 Backtest Results")
    try:
        df = pd.read_csv("data/entry_log.csv")
        total = len(df)
        valid = df[df["Entry Valid"] == True]
        valid_count = len(valid)
        avg_conf = valid["Confidence"].mean()
        st.metric("Total Signals", total)
        st.metric("Valid Entries", valid_count)
        st.metric("Average Confidence", round(avg_conf, 2))
        st.dataframe(valid.sort_values(by="Confidence", ascending=False))
    except Exception as e:
        st.warning(f"Backtest log not found or error reading: {e}")
