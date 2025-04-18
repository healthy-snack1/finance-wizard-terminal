
import streamlit as st
from modules.scanner import run_scanner

st.set_page_config(page_title="Finance Wizard ARK Scanner", layout="wide")

st.title("🔮 Finance Wizard — ARK-Style Swing Scanner")

tabs = st.tabs(["📈 Scanner", "🎯 ARK Entry Guide"])

with tabs[0]:
    run_scanner()

with tabs[1]:
    st.subheader("🎯 When to Enter a Trade — ARK-Inspired Entry Strategy")
    
    st.markdown("""
### ✅ Entry Conditions (Adapted from ARK-style Swing Entries)

**Ideal entry setup happens when:**

- 📊 **Price is pulling back** 3–8% to the rising **EMA20**
- 🔁 Stock is in a **clear uptrend** (Price > EMA20 > EMA50)
- 🔊 **Volume surges** during or right after pullback
- 📈 **Breakout candle appears** (hammer, engulfing, or inside bar)
- 🧠 AI Confidence Score ≥ 65%

---

### 🕰️ Entry Timing Tips

- Enter **at or near** the close of the **confirmation candle**
- Or use a **buy stop slightly above** breakout candle high
- If using a trigger, set at: `Previous High + 1%`

---

### 🧠 Bonus Entry Rules

- ✅ Avoid extended stocks (price making 5-day highs)
- ✅ Favor those near support + trendlines
- 🚀 Look for clean breakouts on relative volume

Use this as your final checklist before entering trades identified by the scanner.
""")
