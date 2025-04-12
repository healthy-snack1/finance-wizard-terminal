
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def run_compounding_simulator():
    st.subheader("💹 Compounding Growth Simulator")

    col1, col2 = st.columns(2)
    with col1:
        starting_balance = st.number_input("Starting Balance ($)", value=2500.0)
        risk_percent = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)
        win_rate = st.slider("Win Rate (%)", 30, 100, 60)
    with col2:
        avg_gain = st.slider("Avg % Gain on Win", 1, 20, 5)
        avg_loss = st.slider("Avg % Loss on Loss", 1, 20, 5)
        trades_per_week = st.slider("Trades per Week", 1, 20, 5)

    weeks = st.slider("Simulation Length (Weeks)", 4, 52, 24)

    if st.button("Simulate Growth"):
        balance = starting_balance
        history = []

        for week in range(1, weeks + 1):
            for _ in range(trades_per_week):
                trade_size = balance * (risk_percent / 100)
                outcome = "win" if pd.Series([1]).sample(p=[win_rate / 100, 1 - win_rate / 100])[0] == 1 else "loss"
                if outcome == "win":
                    balance += trade_size * (avg_gain / 100)
                else:
                    balance -= trade_size * (avg_loss / 100)
            history.append(balance)

        df = pd.DataFrame({"Week": list(range(1, weeks + 1)), "Balance": history})
        st.line_chart(df.set_index("Week"))
        st.success(f"Final Balance after {weeks} weeks: ${balance:,.2f}")
