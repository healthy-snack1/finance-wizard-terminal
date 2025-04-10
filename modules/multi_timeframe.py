from typing import List
import pandas as pd
from utils.fetch_data import fetch_ohlcv_data
from utils.indicators import compute_rsi

def is_signal_confirmed(symbol: str, timeframes: List[str] = ["1m", "5m", "15m"], rsi_threshold: int = 30) -> bool:
    try:
        for tf in timeframes:
            df = fetch_ohlcv_data(symbol, interval=tf)
            if df is None or df.empty:
                print(f"[!] No data for {symbol} on {tf} timeframe")
                return False

            rsi = compute_rsi(df['close'])
            if rsi.iloc[-1] > rsi_threshold:
                print(f"[x] RSI on {tf} = {rsi.iloc[-1]:.2f}, not below {rsi_threshold}")
                return False

        print(f"[✓] Signal confirmed for {symbol} across all timeframes")
        return True
    except Exception as e:
        print(f"[ERROR] in is_signal_confirmed: {e}")
        return False
