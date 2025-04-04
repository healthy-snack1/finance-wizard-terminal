# Streamlit app launcher
import streamlit as st
from modules.scanner import run_scanner
from modules.journal import run_journal

st.set_page_config(page_title='Finance Wizard Swing Terminal', layout='wide')
st.title('📈 Finance Wizard Swing Trade Terminal')

tabs = st.tabs(["🔍 Swing Scanner", "📓 Trade Journal"])

with tabs[0]:
    run_scanner()

with tabs[1]:
    run_journal()
