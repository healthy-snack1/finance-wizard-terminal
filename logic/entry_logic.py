from modules.multi_timeframe import is_signal_confirmed

def should_enter_trade(symbol: str, data: dict, use_multitf: bool = True) -> bool:
    if use_multitf and not is_signal_confirmed(symbol):
        return False

    rsi = data.get('rsi', 100)
    volume = data.get('volume')
    volume_avg = data.get('volume_avg')

    if rsi < 30 and volume > volume_avg:
        return True

    return False
