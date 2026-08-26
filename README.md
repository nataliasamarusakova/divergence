# Event-Driven Coinalyze -> BingX VST Trader

## No lifecycle

This repository intentionally does not use the old 5-snapshot / 5-minute lifecycle.

The flow is:

`Coinalyze discovery -> BingX 1H event -> 15m trigger -> setup -> Telegram -> BingX VST`

### Events

1H:
- Regular Bullish/Bearish RSI divergence
- Regular Bullish/Bearish BingX local CVD divergence
- Volatility squeeze release

15m:
- Execution trigger only

## Telegram

A signal is sent as:

`🚨 LONG SIGNAL`

or

`🔻 SHORT SIGNAL`

The message includes event, price, Coinalyze context, entry, SL, TP, R:R and BingX order status.

`TG_BOT_TOKEN` and `TG_CHAT_IDS` are read from GitHub Actions secrets.

## VST execution

Default configuration is VST:

`BINGX_ENV=vst`
`EXECUTION_ENABLED=true`

The execution adapter submits a MARKET order and attaches one structural STOP_MARKET plus one TAKE_PROFIT_MARKET to that order.

Before using live money, keep `BINGX_ENV=vst` and verify the exact account position mode and contract rules.

## Start locally

```bash
pip install -r requirements.txt
pytest -q
python run_once.py
```

## GitHub Actions

Use `.github/workflows/event-engine.yml`.

Repository secrets:

- `COINALYZE_URL`
- `BINGX_API_KEY`
- `BINGX_SECRET_KEY`
- `TG_BOT_TOKEN`
- `TG_CHAT_IDS`

`TG_CHAT_IDS` accepts comma- or semicolon-separated chat IDs.
