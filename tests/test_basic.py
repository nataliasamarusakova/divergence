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

def test_parse_table_preserves_extra_raw_fields():
    from event_engine.coinalyze import parse_table
    html = '''
    <table><tbody><tr data-coin="BTC">
      <td>#1</td><td><span>Bitcoin</span><span>BTC</span></td>
      <td>100</td><td>1%</td><td>$10M</td><td>$20M</td><td>$5M</td>
      <td>2%</td><td>1M</td><td>3%</td><td>2M</td><td>0.25</td><td>0.01</td>
      <td>0.2</td><td>0.1</td><td>0.001</td><td>0.002</td><td>$1000</td><td>$500</td>
      <td>1.1</td><td>0.5</td><td>50</td><td>20</td><td>10</td><td>0</td>
    </tr></tbody></table>
    '''
    rows = parse_table(html)
    assert len(rows) == 1
    assert rows[0].symbol == 'BTC'
    assert rows[0].price == 100.0
    assert rows[0].raw['mktcap'] == 10_000_000
