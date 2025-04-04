# Streamlit app launcher
import streamlit as st
from modules.scanner import run_scanner
from modules.journal import run_journal
from modules.entry_logic import calculate_trigger_and_flags, log_entry_analysis
from modules.backtest_dashboard import run_backtest_dashboard
from modules.ai_optimizer import update_ai_scoreboard


st.set_page_config(page_title='Finance Wizard Swing Terminal', layout='wide')
st.title('📈 Finance Wizard Swing Trade Terminal')

tabs = st.tabs(["🔍 Swing Scanner", "📓 Trade Journal", "📊 Backtest Results"])


with tabs[0]:
    run_scanner()

with tabs[1]:
    run_journal()

with tabs[2]:
    run_backtest_dashboard()
