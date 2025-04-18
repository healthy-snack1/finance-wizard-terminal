
import pandas as pd
import os
from datetime import datetime

LOG_FILE = "data/entry_log.csv"
MODEL_TRAIN_LOG = "data/ai_scoreboard.csv"

def update_ai_scoreboard():
    if not os.path.exists(LOG_FILE):
        print("No entry log found.")
        return

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        print("Entry log is empty.")
        return

    # Keep only today's entries if it's being called at 5PM daily
    today = datetime.now().strftime("%Y-%m-%d")
    df_today = df[df['Date'] == today]
    if df_today.empty:
        print("No new data to log for today.")
        return

    # Create or append to AI scoreboard
    if os.path.exists(MODEL_TRAIN_LOG):
        scoreboard_df = pd.read_csv(MODEL_TRAIN_LOG)
    else:
        scoreboard_df = pd.DataFrame(columns=["Date", "Ticker", "Confidence", "Entry Valid", "Reasons"])

    for _, row in df_today.iterrows():
        entry = {
            "Date": row["Date"],
            "Ticker": row["Ticker"],
            "Confidence": row["Confidence"],
            "Entry Valid": row["Entry Valid"],
            "Reasons": row["Reasons"]
        }
        scoreboard_df = pd.concat([scoreboard_df, pd.DataFrame([entry])], ignore_index=True)

    scoreboard_df.to_csv(MODEL_TRAIN_LOG, index=False)
    print(f"{len(df_today)} entries logged to AI scoreboard.")
