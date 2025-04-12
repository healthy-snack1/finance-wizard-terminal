
import pandas as pd
import os
from datetime import datetime

def update_ai_scoreboard(ticker, prediction, confidence, triggered, log_file='data/ai_scoreboard.csv'):
    now = datetime.now().strftime("%Y-%m-%d")
    row = [now, ticker, prediction, confidence, triggered]
    columns = ["Date", "Ticker", "Prediction", "Confidence", "Triggered"]

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
    else:
        df = pd.DataFrame(columns=columns)

    df.loc[len(df)] = row
    df.to_csv(log_file, index=False)
