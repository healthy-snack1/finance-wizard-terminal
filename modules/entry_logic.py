
import os
import pandas as pd
from datetime import datetime

def calculate_trigger_and_flags(latest, df, avg_volume):
    price = latest['Close']
    trigger_price = round(price * 1.01, 2)  # default trigger 1% above current close
    reasons = []

    if latest['Close'] > latest['EMA20']:
        reasons.append("Price > EMA20")
    if latest['MACD'] > latest['Signal']:
        reasons.append("MACD crossover")
    if latest['RSI'] > 60:
        reasons.append("RSI > 60")
    if latest['Volume'] > avg_volume:
        reasons.append("Volume Surge")

    entry_valid = len(reasons) >= 2
    return trigger_price, entry_valid, ", ".join(reasons)

def log_entry_analysis(ticker, trigger_price, entry_valid, reasons, confidence):
    path = "data/entry_log.csv"
    date = datetime.now().strftime("%Y-%m-%d")

    entry = {
        "Date": date,
        "Ticker": ticker,
        "Trigger Price": trigger_price,
        "Entry Valid": entry_valid,
        "Reasons": reasons,
        "Confidence": confidence
    }

    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=list(entry.keys()))

    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(path, index=False)
