from __future__ import annotations

from dataclasses import dataclass
import os


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1','true','yes','y','on'}


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


@dataclass(frozen=True)
class Config:
    coinalyze_url: str = os.getenv(
        'COINALYZE_URL',
        'https://coinalyze.net/?order_by=price_24hour_pchange&order_dir=asc',
    )
    min_volume_24h: float = _float('MIN_VOLUME_24H', 1_000_000.0)
    min_open_interest: float = _float('MIN_OPEN_INTEREST', 500_000.0)
    max_candidates: int = _int('MAX_CANDIDATES', 40)

    bingx_env: str = os.getenv('BINGX_ENV', 'vst')
    bingx_api_key: str = os.getenv('BINGX_API_KEY', '')
    bingx_secret_key: str = os.getenv('BINGX_SECRET_KEY', '')
    recv_window: int = _int('BINGX_RECV_WINDOW', 5000)
    kline_limit_1h: int = _int('KLINE_LIMIT_1H', 250)
    kline_limit_15m: int = _int('KLINE_LIMIT_15M', 250)

    # Frozen divergence configuration.
    pivot_left: int = 3
    pivot_right: int = 2
    min_bars_between: int = 5
    max_bars_between: int = 35
    min_price_delta_atr: float = 0.25

    # 1H volatility squeeze.
    bb_length: int = 20
    bb_std: float = 2.0
    kc_length: int = 20
    kc_atr_mult: float = 1.5
    squeeze_ratio_max: float = 0.80

    # Event -> trigger.
    max_event_age_min: int = _int('MAX_EVENT_AGE_MIN', 45)
    require_15m_trigger: bool = _bool('REQUIRE_15M_TRIGGER', True)
    require_cvd_confirmation: bool = _bool('REQUIRE_CVD_CONFIRMATION', True)

    # Risk / VST.
    execution_enabled: bool = _bool('EXECUTION_ENABLED', True)
    execution_mode: str = os.getenv('EXECUTION_MODE', 'vst')
    position_mode: str = os.getenv('BINGX_POSITION_MODE', 'HEDGE')  # HEDGE or ONE_WAY
    margin_usdt: float = _float('BINGX_MARGIN_USDT', 1.0)
    leverage: float = _float('BINGX_LEVERAGE', 10.0)
    sl_atr_buffer: float = _float('SL_ATR_BUFFER', 0.50)
    target_r_multiple: float = _float('TARGET_R_MULTIPLE', 2.0)
    max_trades_per_cycle: int = _int('MAX_TRADES_PER_CYCLE', 1)
    max_daily_trades: int = _int('MAX_DAILY_TRADES', 5)

    # Telegram.
    telegram_enabled: bool = _bool('TG_ENABLED', True)
    telegram_bot_token: str = os.getenv('TG_BOT_TOKEN', '')
    telegram_chat_ids: str = os.getenv('TG_CHAT_IDS', os.getenv('TG_CHAT_ID', ''))


CONFIG = Config()
