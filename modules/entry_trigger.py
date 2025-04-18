
import yfinance as yf
import pandas as pd

def get_entry_trigger(ticker):
    try:
        df = yf.Ticker(ticker).history(period="3mo")
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

        latest = df.iloc[-1]
        prev_high = df['Close'].rolling(window=5).max().iloc[-2]
        price = latest['Close']
        ema20 = latest['EMA20']
        ema50 = latest['EMA50']

        # Validate ARK-style setup
        if price > ema20 > ema50:
            trigger_price = round(prev_high * 1.01, 2)
            return {
                "Ticker": ticker,
                "Current Price": round(price, 2),
                "EMA20": round(ema20, 2),
                "EMA50": round(ema50, 2),
                "Previous 5D High": round(prev_high, 2),
                "Suggested Entry Trigger": trigger_price,
                "Notes": "Trigger set 1% above recent high"
            }
        else:
            return {
                "Ticker": ticker,
                "Current Price": round(price, 2),
                "EMA20": round(ema20, 2),
                "EMA50": round(ema50, 2),
                "Suggested Entry Trigger": "N/A",
                "Notes": "Stock not in confirmed uptrend (Price > EMA20 > EMA50)"
            }
    except Exception as e:
        return {
            "Ticker": ticker,
            "Error": str(e)
        }
