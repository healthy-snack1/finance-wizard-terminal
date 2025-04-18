
import streamlit as st
from modules.scanner import run_scanner

st.set_page_config(page_title="Finance Wizard ARK Scanner", layout="wide")

st.title("🔮 Finance Wizard — ARK-Style Swing Scanner")
run_scanner()
