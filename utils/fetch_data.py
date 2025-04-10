import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_ohlcv_data(symbol: str, interval: str = "5m", length: int = 100) -> pd.DataFrame:
    now = datetime.now()
    dates = [now - timedelta(minutes=5 * i) for i in range(length)][::-1]
    close = np.random.normal(loc=100, scale=2, size=length)
    volume = np.random.normal(loc=1_000_000, scale=100_000, size=length)

    return pd.DataFrame({
        "datetime": dates,
        "close": close,
        "volume": volume
    })
