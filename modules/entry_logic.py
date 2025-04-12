
import pandas as pd
import numpy as np
from datetime import datetime
import os

def calculate_trigger_and_flags(latest, df, avg_volume=None):
    trigger_price = round(latest['Close'] * 1.01, 2)
    entry_valid = False
    reasons = []

    if latest['Close'] > latest['EMA20']:
        entry_valid = True
        reasons.append("Close > EMA20")

    if latest['MACD'] > latest['Signal']:
        reasons.append("MACD crossover")

    if latest['RSI'] > 60:
        reasons.append("RSI > 60")

    if avg_volume and latest['Volume'] >= 1.5 * avg_volume:
        reasons.append("Volume surge")

    return trigger_price, entry_valid, ", ".join(reasons)

def log_entry_analysis(ticker, trigger_price, entry_valid, reasons, confidence, log_file='data/entry_log.csv'):
    now = datetime.now().strftime("%Y-%m-%d")
    row = [now, ticker, trigger_price, entry_valid, reasons, confidence]
    columns = ["Date", "Ticker", "Trigger Price", "Entry Valid", "Reasons", "Confidence"]

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
    else:
        df = pd.DataFrame(columns=columns)

    df.loc[len(df)] = row
    df.to_csv(log_file, index=False)
