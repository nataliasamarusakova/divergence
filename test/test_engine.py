from event_engine.coinalyze import parse_number
from event_engine.signals import build_15m_trigger
import pandas as pd

def test_parse_number():
    assert parse_number("1.2M") == 1_200_000
    assert parse_number("$5,000") == 5000

def test_trigger_long():
    df=pd.DataFrame({"high":[100,105],"low":[95,100],"close":[99,106]})
    assert build_15m_trigger(df,"LONG") is True

def test_trigger_short():
    df=pd.DataFrame({"high":[105,104],"low":[100,95],"close":[101,94]})
    assert build_15m_trigger(df,"SHORT") is True

def test_trigger_none():
    df=pd.DataFrame({"high":[100,105],"low":[95,100],"close":[99,99.5]})
    assert build_15m_trigger(df,"LONG") is False


def test_telegram_format_short_and_long():
    from event_engine.telegram import format_signal
    long_msg = format_signal({
        'direction': 'LONG', 'symbol': 'SOL-USDT', 'event_type': 'REGULAR_BULLISH_RSI',
        'timeframe': '1h', 'timestamps': {'detected_at_ts': 1},
        'event_fact': {'detection_close_price': 100},
    }, execution={'mode': 'vst', 'status': 'opened', 'order_id': '123'})
    short_msg = format_signal({
        'direction': 'SHORT', 'symbol': 'SOL-USDT', 'event_type': 'REGULAR_BEARISH_RSI',
        'timeframe': '1h', 'timestamps': {'detected_at_ts': 1},
        'event_fact': {'detection_close_price': 100},
    })
    assert '🚨 LONG SIGNAL' in long_msg
    assert '🔻 SHORT SIGNAL' in short_msg
    assert 'opened' in long_msg
