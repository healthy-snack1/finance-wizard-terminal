import streamlit as st
from modules.scanner import run_scanner
from modules.journal import run_journal
from modules.backtest_dashboard import run_backtest_dashboard
from modules.entry_check import run_entry_check

st.set_page_config(page_title='Finance Wizard Swing Terminal', layout='wide')
st.title('📈 Finance Wizard Swing Trade Terminal')

tabs = st.tabs(["🔍 Swing Scanner", "📓 Trade Journal", "📊 Backtest Results", "📥 Entry Signal Checker"])

with tabs[0]:
    run_scanner()

with tabs[1]:
    run_journal()

with tabs[2]:
    run_backtest_dashboard()

with tabs[3]:
    run_entry_check()
