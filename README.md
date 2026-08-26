# Event Driven Coinalyze -> BingX VST

Pipeline: Coinalyze browser discovery + pagination -> BingX contract mapping -> 1H RSI/CVD divergence or 1H volatility squeeze -> 15m trigger -> Telegram -> BingX VST execution.

GitHub Actions secrets:
COINALYZE_URL, COINALYZE_P_SID, COINALYZE_CHAT_SID, BINGX_API_KEY, BINGX_SECRET_KEY, TG_BOT_TOKEN, TG_CHAT_IDS.

The workflow explicitly installs the Playwright Chromium executable before running tests/engine.
