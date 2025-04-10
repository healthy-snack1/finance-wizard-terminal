import streamlit as st
import pandas as pd
import os

JOURNAL_PATH = "data/journal.csv"
os.makedirs("data", exist_ok=True)

def run_journal():
    st.header("📓 Trade Journal")

    if os.path.exists(JOURNAL_PATH):
        df = pd.read_csv(JOURNAL_PATH)
        st.dataframe(df)
    else:
        st.info("No journal entries yet.")

    st.subheader("➕ Add Trade Entry")
    symbol = st.text_input("Symbol")
    result = st.selectbox("Result", ["Win", "Loss", "Break-even"])
    notes = st.text_area("Notes")

    if st.button("Add Entry"):
        entry = pd.DataFrame([[symbol, result, notes]], columns=["Symbol", "Result", "Notes"])
        if os.path.exists(JOURNAL_PATH):
            existing = pd.read_csv(JOURNAL_PATH)
            entry = pd.concat([existing, entry], ignore_index=True)
        entry.to_csv(JOURNAL_PATH, index=False)
        st.success("✅ Entry added!")
