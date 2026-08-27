from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n,
    ).mean()

    zero_loss = (avg_loss == 0) & (avg_gain > 0)
    zero_both = (avg_loss == 0) & (avg_gain == 0)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(~zero_loss, 100.0)
    rsi = rsi.where(~zero_both, 50.0)

    valid_warmup = avg_gain.notna() & avg_loss.notna()

    return rsi.where(valid_warmup, np.nan)


def _bbands(
    series: pd.Series,
    n: int = 20,
    std: float = 2.0,
):
    mid = series.rolling(
        n,
        min_periods=n,
    ).mean()

    sd = series.rolling(
        n,
        min_periods=n,
    ).std(ddof=0)

    return (
        mid + std * sd,
        mid,
        mid - std * sd,
    )


def _atr(
    df: pd.DataFrame,
    n: int = 14,
) -> pd.Series:
    tr = pd.concat(
        [
            df.high - df.low,
            (df.high - df.close.shift()).abs(),
            (df.low - df.close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(
        n,
        min_periods=n,
    ).mean()


def _pivots(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 2,
):
    lows = []
    highs = []

    for i in range(left, len(df) - right):
        if (
            float(df.low.iloc[i])
            <= float(df.low.iloc[i - left:i].min())
            and float(df.low.iloc[i])
            < float(df.low.iloc[i + 1:i + right + 1].min())
        ):
            lows.append(i)

        if (
            float(df.high.iloc[i])
            >= float(df.high.iloc[i - left:i].max())
            and float(df.high.iloc[i])
            > float(df.high.iloc[i + 1:i + right + 1].max())
        ):
            highs.append(i)

    return lows, highs


def _event_id(
    symbol: str,
    tf: str,
    typ: str,
    p1_ts: int,
    p2_ts: int,
) -> str:
    fp = (
        f"{symbol}:{tf}:{typ}:"
        f"{int(p1_ts)}:{int(p2_ts)}"
    )

    return (
        "EVT_"
        + hashlib.sha256(
            fp.encode()
        ).hexdigest()[:16].upper()
    )


def add_cvd(
    df: pd.DataFrame,
) -> pd.DataFrame:
    work = df.copy()

    if "bar_delta_usdt" not in work.columns:
        work["bingx_cvd"] = float("nan")
        work["cvd_segment_id"] = 0
        return work

    if "taker_flow_valid" not in work.columns:
        work["taker_flow_valid"] = False

    work["taker_flow_valid"] = (
        work["taker_flow_valid"]
        .fillna(False)
        .astype(bool)
    )

    work["cvd_segment_id"] = (
        ~work["taker_flow_valid"]
    ).cumsum()

    work["bingx_cvd"] = float("nan")

    for _, idx in work.groupby(
        "cvd_segment_id",
        sort=False,
    ).groups.items():

        valid_idx = [
            i
            for i in idx
            if bool(
                work.at[
                    i,
                    "taker_flow_valid",
                ]
            )
            and pd.notna(
                work.at[
                    i,
                    "bar_delta_usdt",
                ]
            )
        ]

        if not valid_idx:
            continue

        work.loc[
            valid_idx,
            "bingx_cvd",
        ] = work.loc[
            valid_idx,
            "bar_delta_usdt",
        ].cumsum()

    return work


def detect_divergences(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h",
    left: int = 3,
    right: int = 2,
    min_bars: int = 5,
    max_bars: int = 35,
    min_delta_atr: float = 0.25,
) -> list[dict[str, Any]]:

    if len(df) < 60:
        return []

    work = df.copy()

    work["rsi"] = _rsi(
        work["close"],
        14,
    )

    work["atr"] = _atr(
        work,
        14,
    )

    lows, highs = _pivots(
        work,
        left,
        right,
    )

    events: list[dict[str, Any]] = []

    def emit_divergence(
        p1i: int,
        p2i: int,
        indicator: str,
        is_low: bool,
    ):
        bars = p2i - p1i

        if not (
            min_bars
            <= bars
            <= max_bars
        ):
            return

        atr_value = work.atr.iloc[p2i]

        atr = (
            float(atr_value)
            if pd.notna(atr_value)
            else 0.0
        )

        if atr <= 0:
            return

        p1v = work[indicator].iloc[p1i]
        p2v = work[indicator].iloc[p2i]

        if pd.isna(p1v) or pd.isna(p2v):
            return

        detected = p2i + right

        if detected >= len(work):
            return

        if indicator == "bingx_cvd":
            span = work.iloc[
                p1i:detected + 1
            ]

            if (
                not span[
                    "bar_delta_usdt"
                ].notna().all()
                or work.cvd_segment_id.iloc[p1i]
                != work.cvd_segment_id.iloc[detected]
            ):
                return

        price_col = (
            "low"
            if is_low
            else "high"
        )

        p1_price = float(
            work[
                price_col
            ].iloc[p1i]
        )

        p2_price = float(
            work[
                price_col
            ].iloc[p2i]
        )

        price_delta_atr = (
            abs(
                p2_price - p1_price
            )
            / atr
        )

        if (
            price_delta_atr
            < min_delta_atr
        ):
            return

        typ = None
        direction = None

        if is_low:

            if (
                p2_price < p1_price
                and p2v > p1v
            ):
                typ = (
                    "REGULAR_BULLISH_"
                    f"{indicator.upper()}"
                )
                direction = "LONG"

            elif (
                p2_price > p1_price
                and p2v < p1v
            ):
                typ = (
                    "HIDDEN_BULLISH_"
                    f"{indicator.upper()}"
                )
                direction = "LONG"

        else:

            if (
                p2_price > p1_price
                and p2v < p1v
            ):
                typ = (
                    "REGULAR_BEARISH_"
                    f"{indicator.upper()}"
                )
                direction = "SHORT"

            elif (
                p2_price < p1_price
                and p2v > p1v
            ):
                typ = (
                    "HIDDEN_BEARISH_"
                    f"{indicator.upper()}"
                )
                direction = "SHORT"

        if not typ or not direction:
            return

        p1_ts = int(
            work.close_time.iloc[p1i]
        )

        p2_ts = int(
            work.close_time.iloc[p2i]
        )

        detected_ts = int(
            work.close_time.iloc[detected]
        )

        events.append(
            {
                "event_id": _event_id(
                    symbol,
                    timeframe,
                    typ,
                    p1_ts,
                    p2_ts,
                ),
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "event_type": typ,
                "timestamps": {
                    "pivot_1_ts": p1_ts,
                    "pivot_2_ts": p2_ts,
                    "detected_at_ts": detected_ts,
                },
                "event_fact": {
                    "detection_close_price": float(
                        work.close.iloc[detected]
                    ),
                    "p1_price": p1_price,
                    "p2_price": p2_price,
                    "p1_indicator": float(p1v),
                    "p2_indicator": float(p2v),
                    "bars_between": bars,
                    "price_delta_atr": float(
                        price_delta_atr
                    ),
                },
            }
        )

    def scan_pivots(
        pivots: list[int],
        is_low: bool,
    ):
        num_piv = len(pivots)

        for i in range(num_piv):

            for step in (1, 2):

                if i + step >= num_piv:
                    continue

                p1 = pivots[i]
                p2 = pivots[i + step]

                emit_divergence(
                    p1,
                    p2,
                    "rsi",
                    is_low,
                )

                emit_divergence(
                    p1,
                    p2,
                    "bingx_cvd",
                    is_low,
                )

    if len(lows) >= 2:
        scan_pivots(
            lows,
            is_low=True,
        )

    if len(highs) >= 2:
        scan_pivots(
            highs,
            is_low=False,
        )

    return events


def detect_squeeze_release(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h",
    min_squeeze_bars: int = 3,
) -> list[dict[str, Any]]:

    if len(df) < 40:
        return []

    b = df.copy()

    bb_u, mid, bb_l = _bbands(
        b.close,
        20,
        2.0,
    )

    atr = _atr(
        b,
        20,
    )

    kc_u = mid + 1.5 * atr
    kc_l = mid - 1.5 * atr

    in_sq = (
        (bb_u <= kc_u)
        & (bb_l >= kc_l)
    )

    if (
        len(b)
        < min_squeeze_bars + 2
        or pd.isna(bb_u.iloc[-1])
        or pd.isna(kc_u.iloc[-1])
    ):
        return []

    is_currently_in = bool(
        in_sq.iloc[-1]
    )

    was_previously_in = bool(
        in_sq.iloc[-2]
    )

    if (
        is_currently_in
        or not was_previously_in
    ):
        return []

    sq_duration = 0
    idx = len(in_sq) - 2

    while (
        idx >= 0
        and in_sq.iloc[idx]
    ):
        sq_duration += 1
        idx -= 1

    if (
        sq_duration
        < min_squeeze_bars
    ):
        return []

    close_val = float(
        b.close.iloc[-1]
    )

    kc_u_val = float(
        kc_u.iloc[-1]
    )

    kc_l_val = float(
        kc_l.iloc[-1]
    )

    if close_val > kc_u_val:
        direction = "LONG"

    elif close_val < kc_l_val:
        direction = "SHORT"

    else:
        return []

    ts = int(
        b.close_time.iloc[-1]
    )

    typ = (
        "VOLATILITY_SQUEEZE_RELEASE"
    )

    bb_w = float(
        bb_u.iloc[-1]
        - bb_l.iloc[-1]
    )

    kc_w = float(
        kc_u.iloc[-1]
        - kc_l.iloc[-1]
    )

    compression_ratio = (
        float(bb_w / kc_w)
        if kc_w > 0
        else 1.0
    )

    return [
        {
            "event_id": _event_id(
                symbol,
                timeframe,
                typ,
                ts,
                ts,
            ),
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "event_type": typ,
            "timestamps": {
                "pivot_1_ts": ts,
                "pivot_2_ts": ts,
                "detected_at_ts": ts,
            },
            "event_fact": {
                "detection_close_price": close_val,
                "bb_width": bb_w,
                "kc_width": kc_w,
                "squeeze_duration_bars": sq_duration,
                "compression_ratio": compression_ratio,
            },
        }
    ]


def diagnose_15m_trigger(
    df15: pd.DataFrame,
    direction: str,
    min_vol_mult: float = 1.0,
) -> dict[str, Any]:
    """
    Диагностический вариант 15M trigger.

    Торговую логику НЕ меняет.
    Возвращает точную причину PASS/FAIL.
    """

    result = {
        "ok": False,
        "reason": None,
        "direction": str(direction).upper(),
        "breakout_pass": False,
        "volume_pass": True,
        "data_pass": True,
        "direction_pass": True,
        "previous_high": None,
        "previous_low": None,
        "current_close": None,
        "current_volume": None,
        "volume_sma20": None,
        "volume_ratio": None,
    }

    if not isinstance(
        df15,
        pd.DataFrame,
    ) or len(df15) < 2:

        result["ok"] = False
        result["reason"] = (
            "insufficient_data"
        )
        result["data_pass"] = False
        return result

    d = str(direction).upper()

    if d not in {
        "LONG",
        "SHORT",
    }:
        result["ok"] = False
        result["reason"] = (
            "invalid_direction"
        )
        result["direction_pass"] = False
        return result

    h = df15.iloc[-1]
    p = df15.iloc[-2]

    try:
        current_close = float(
            h["close"]
        )

        previous_high = float(
            p["high"]
        )

        previous_low = float(
            p["low"]
        )

        current_volume = (
            float(h["volume"])
            if "volume" in df15.columns
            else None
        )

    except (
        TypeError,
        ValueError,
        KeyError,
    ) as exc:

        result["ok"] = False
        result["reason"] = (
            "invalid_15m_data"
        )
        result["data_pass"] = False
        result["error"] = str(exc)
        return result

    result[
        "previous_high"
    ] = previous_high

    result[
        "previous_low"
    ] = previous_low

    result[
        "current_close"
    ] = current_close

    result[
        "current_volume"
    ] = current_volume

    if d == "LONG":
        breakout_pass = (
            current_close
            > previous_high
        )
    else:
        breakout_pass = (
            current_close
            < previous_low
        )

    result[
        "breakout_pass"
    ] = bool(breakout_pass)

    if not breakout_pass:
        result["ok"] = False
        result["reason"] = (
            "breakout_failed"
        )
        return result

    # Same volume logic as before.
    if (
        "volume" in df15.columns
        and len(df15) >= 20
        and min_vol_mult > 0
    ):

        volume_series = pd.to_numeric(
            df15["volume"],
            errors="coerce",
        )

        vol_sma = volume_series.iloc[
            -21:-1
        ].mean()

        if (
            pd.notna(vol_sma)
            and float(vol_sma) > 0
        ):

            vol_sma = float(vol_sma)

            result[
                "volume_sma20"
            ] = vol_sma

            if current_volume is not None:
                volume_ratio = (
                    current_volume
                    / vol_sma
                )

                result[
                    "volume_ratio"
                ] = float(volume_ratio)

                if (
                    current_volume
                    < vol_sma
                    * min_vol_mult
                ):
                    result[
                        "volume_pass"
                    ] = False

                    result["ok"] = False
                    result["reason"] = (
                        "volume_failed"
                    )

                    return result

    result["volume_pass"] = True
    result["ok"] = True
    result["reason"] = "passed"

    return result


def build_15m_trigger(
    df15: pd.DataFrame,
    direction: str,
    min_vol_mult: float = 1.0,
) -> bool:
    """
    Compatibility wrapper.

    Старый контракт функции сохраняется:
    возвращает только bool.

    Внутри используется тот же самый
    диагностический механизм.
    """

    diagnostic = diagnose_15m_trigger(
        df15,
        direction,
        min_vol_mult,
    )

    return bool(
        diagnostic["ok"]
    )


def check_btc_regime(
    btc_1h_df: pd.DataFrame,
    direction: str,
) -> tuple[bool, str]:

    if len(btc_1h_df) < 5:
        return True, "INSUFFICIENT_DATA"

    close = btc_1h_df["close"]

    last_close = float(
        close.iloc[-1]
    )

    prev_1h = float(
        close.iloc[-2]
    )

    prev_4h = float(
        close.iloc[-5]
    )

    if prev_1h <= 0 or prev_4h <= 0:
        return False, "INVALID_BTC_PRICE"

    chg_1h_pct = (
        (
            last_close
            - prev_1h
        )
        / prev_1h
    ) * 100.0

    chg_4h_pct = (
        (
            last_close
            - prev_4h
        )
        / prev_4h
    ) * 100.0

    d = str(
        direction
    ).upper()

    if d == "LONG":

        if chg_1h_pct < -1.2:
            return (
                False,
                f"BTC_DUMPING_1H ({chg_1h_pct:.2f}%)",
            )

        if chg_4h_pct < -2.5:
            return (
                False,
                f"BTC_DUMPING_4H ({chg_4h_pct:.2f}%)",
            )

    elif d == "SHORT":

        if chg_1h_pct > 1.5:
            return (
                False,
                f"BTC_PUMPING_1H (+{chg_1h_pct:.2f}%)",
            )

        if chg_4h_pct > 3.0:
            return (
                False,
                f"BTC_PUMPING_4H (+{chg_4h_pct:.2f}%)",
            )

    return True, "OK"
