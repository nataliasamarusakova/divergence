from __future__ import annotations

import hashlib
import math
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

    zero_loss = (
        (avg_loss == 0)
        & (avg_gain > 0)
    )

    zero_both = (
        (avg_loss == 0)
        & (avg_gain == 0)
    )

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    rsi = 100.0 - (
        100.0 / (1.0 + rs)
    )

    rsi = rsi.where(
        ~zero_loss,
        100.0,
    )

    rsi = rsi.where(
        ~zero_both,
        50.0,
    )

    valid_warmup = (
        avg_gain.notna()
        & avg_loss.notna()
    )

    return rsi.where(
        valid_warmup,
        np.nan,
    )


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
    ).std(
        ddof=0
    )

    upper = mid + std * sd
    lower = mid - std * sd

    return upper, mid, lower


def _atr(
    df: pd.DataFrame,
    n: int = 14,
) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - df["close"].shift()
            ).abs(),
            (
                df["low"]
                - df["close"].shift()
            ).abs(),
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
    lows: list[int] = []
    highs: list[int] = []

    for i in range(
        left,
        len(df) - right,
    ):
        current_low = float(
            df["low"].iloc[i]
        )

        previous_lows = df[
            "low"
        ].iloc[
            i - left:i
        ]

        next_lows = df[
            "low"
        ].iloc[
            i + 1:i + right + 1
        ]

        current_high = float(
            df["high"].iloc[i]
        )

        previous_highs = df[
            "high"
        ].iloc[
            i - left:i
        ]

        next_highs = df[
            "high"
        ].iloc[
            i + 1:i + right + 1
        ]

        if (
            current_low
            <= float(previous_lows.min())
            and current_low
            < float(next_lows.min())
        ):
            lows.append(i)

        if (
            current_high
            >= float(previous_highs.max())
            and current_high
            > float(next_highs.max())
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
        ] = (
            work.loc[
                valid_idx,
                "bar_delta_usdt",
            ]
            .cumsum()
        )

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

    required = {
        "close",
        "high",
        "low",
        "close_time",
    }

    if not required.issubset(
        df.columns
    ):
        return []

    work = df.copy()

    for col in (
        "close",
        "high",
        "low",
        "close_time",
    ):
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

    work = work.dropna(
        subset=[
            "close",
            "high",
            "low",
            "close_time",
        ]
    ).reset_index(
        drop=True
    )

    if len(work) < 60:
        return []

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
    ) -> None:

        bars = p2i - p1i

        if not (
            min_bars
            <= bars
            <= max_bars
        ):
            return

        atr_value = work[
            "atr"
        ].iloc[p2i]

        if pd.isna(atr_value):
            return

        atr = float(
            atr_value
        )

        if (
            not math.isfinite(atr)
            or atr <= 0
        ):
            return

        if indicator not in work.columns:
            return

        p1v = work[
            indicator
        ].iloc[p1i]

        p2v = work[
            indicator
        ].iloc[p2i]

        if (
            pd.isna(p1v)
            or pd.isna(p2v)
        ):
            return

        detected = p2i + right

        if detected >= len(work):
            return

        if indicator == "bingx_cvd":

            if (
                "bar_delta_usdt"
                not in work.columns
            ):
                return

            span = work.iloc[
                p1i:detected + 1
            ]

            if not span[
                "bar_delta_usdt"
            ].notna().all():
                return

            if (
                work[
                    "cvd_segment_id"
                ].iloc[p1i]
                !=
                work[
                    "cvd_segment_id"
                ].iloc[detected]
            ):
                return

        price_column = (
            "low"
            if is_low
            else "high"
        )

        p1_price = float(
            work[
                price_column
            ].iloc[p1i]
        )

        p2_price = float(
            work[
                price_column
            ].iloc[p2i]
        )

        price_delta_atr = (
            abs(
                p2_price
                - p1_price
            )
            / atr
        )

        if (
            not math.isfinite(
                price_delta_atr
            )
            or price_delta_atr
            < min_delta_atr
        ):
            return

        typ = None
        direction = None

        if is_low:

            if (
                p2_price < p1_price
                and float(p2v)
                > float(p1v)
            ):
                typ = (
                    "REGULAR_BULLISH_"
                    + indicator.upper()
                )
                direction = "LONG"

            elif (
                p2_price > p1_price
                and float(p2v)
                < float(p1v)
            ):
                typ = (
                    "HIDDEN_BULLISH_"
                    + indicator.upper()
                )
                direction = "LONG"

        else:

            if (
                p2_price > p1_price
                and float(p2v)
                < float(p1v)
            ):
                typ = (
                    "REGULAR_BEARISH_"
                    + indicator.upper()
                )
                direction = "SHORT"

            elif (
                p2_price < p1_price
                and float(p2v)
                > float(p1v)
            ):
                typ = (
                    "HIDDEN_BEARISH_"
                    + indicator.upper()
                )
                direction = "SHORT"

        if not typ or not direction:
            return

        p1_ts = int(
            work[
                "close_time"
            ].iloc[p1i]
        )

        p2_ts = int(
            work[
                "close_time"
            ].iloc[p2i]
        )

        detected_ts = int(
            work[
                "close_time"
            ].iloc[detected]
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
                        work[
                            "close"
                        ].iloc[detected]
                    ),
                    "p1_price": p1_price,
                    "p2_price": p2_price,
                    "p1_indicator": float(
                        p1v
                    ),
                    "p2_indicator": float(
                        p2v
                    ),
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
    ) -> None:

        num_piv = len(
            pivots
        )

        for i in range(
            num_piv
        ):

            for step in (
                1,
                2,
            ):

                if (
                    i + step
                    >= num_piv
                ):
                    continue

                p1 = pivots[i]
                p2 = pivots[
                    i + step
                ]

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

    required = {
        "close",
        "high",
        "low",
        "close_time",
    }

    if not required.issubset(
        df.columns
    ):
        return []

    b = df.copy()

    for col in (
        "close",
        "high",
        "low",
        "close_time",
    ):
        b[col] = pd.to_numeric(
            b[col],
            errors="coerce",
        )

    b = b.dropna(
        subset=[
            "close",
            "high",
            "low",
            "close_time",
        ]
    ).reset_index(
        drop=True
    )

    if len(b) < 40:
        return []

    bb_u, mid, bb_l = _bbands(
        b["close"],
        20,
        2.0,
    )

    atr = _atr(
        b,
        20,
    )

    kc_u = (
        mid
        + 1.5 * atr
    )

    kc_l = (
        mid
        - 1.5 * atr
    )

    in_sq = (
        (bb_u <= kc_u)
        & (bb_l >= kc_l)
    )

    if len(b) < (
        min_squeeze_bars + 2
    ):
        return []

    if (
        pd.isna(
            bb_u.iloc[-1]
        )
        or pd.isna(
            kc_u.iloc[-1]
        )
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

    idx = (
        len(in_sq) - 2
    )

    while (
        idx >= 0
        and bool(
            in_sq.iloc[idx]
        )
    ):
        sq_duration += 1
        idx -= 1

    if (
        sq_duration
        < min_squeeze_bars
    ):
        return []

    close_val = float(
        b["close"].iloc[-1]
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
        b["close_time"].iloc[-1]
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
        bb_w / kc_w
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
                "compression_ratio": float(
                    compression_ratio
                ),
            },
        }
    ]


def diagnose_15m_trigger(
    df15: pd.DataFrame,
    direction: str,
    event_detected_at_ts: int | None = None,
    max_trigger_delay_min: float = 60.0,
    min_vol_mult: float = 1.05,
) -> dict[str, Any]:
    """
    15M trigger с двумя режимами.

    Без event_detected_at_ts:
        legacy-режим. Проверяется последняя свеча
        относительно предыдущей.

    С event_detected_at_ts:
        event-aware режим. Проверяются только
        закрытия после момента события в пределах
        max_trigger_delay_min.
    """

    result: dict[str, Any] = {
        "ok": False,
        "reason": None,
        "direction": str(direction).upper(),
        "event_detected_at_ts": event_detected_at_ts,
        "max_trigger_delay_min": float(
            max_trigger_delay_min
        ),
        "bars_after_event": 0,
        "bars_considered": 0,
        "trigger_bar_close_ts": None,
        "trigger_delay_min": None,
        "previous_high": None,
        "previous_low": None,
        "current_close": None,
        "current_volume": None,
        "volume_sma20": None,
        "volume_ratio": None,
        "breakout_pass": False,
        "volume_pass": True,
        "data_pass": True,
        "direction_pass": True,
        "scanned_bar_details": [],
    }

    if not isinstance(
        df15,
        pd.DataFrame,
    ):
        result["reason"] = "invalid_15m_data"
        result["data_pass"] = False
        return result

    if len(df15) < 2:
        result["reason"] = "insufficient_data"
        result["data_pass"] = False
        return result

    d = str(
        direction
    ).upper()

    if d not in {"LONG", "SHORT"}:
        result["reason"] = "invalid_direction"
        result["direction_pass"] = False
        return result

    required = {
        "close",
        "high",
        "low",
    }

    if event_detected_at_ts is not None:
        required.add("close_time")

    missing = required - set(
        df15.columns
    )

    if missing:
        result["reason"] = "invalid_15m_data"
        result["data_pass"] = False
        result["error"] = (
            "missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )
        return result

    work = df15.copy()

    for col in (
        "close",
        "high",
        "low",
    ):
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

    if "volume" in work.columns:
        work["volume"] = pd.to_numeric(
            work["volume"],
            errors="coerce",
        )

    if "close_time" in work.columns:
        work["close_time"] = pd.to_numeric(
            work["close_time"],
            errors="coerce",
        )

    work = work.dropna(
        subset=[
            "close",
            "high",
            "low",
        ]
    ).reset_index(
        drop=True
    )

    if len(work) < 2:
        result["reason"] = "insufficient_data"
        result["data_pass"] = False
        return result

    # ============================================================
    # LEGACY MODE
    # ============================================================

    if event_detected_at_ts is None:

        i = len(work) - 1

        h = work.iloc[i]
        p = work.iloc[i - 1]

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
        except (
            TypeError,
            ValueError,
        ):
            result["reason"] = (
                "invalid_15m_data"
            )
            result["data_pass"] = False
            return result

        current_volume = None

        if (
            "volume" in work.columns
            and pd.notna(
                h["volume"]
            )
        ):
            try:
                current_volume = float(
                    h["volume"]
                )
            except (
                TypeError,
                ValueError,
            ):
                current_volume = None

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

        volume_sma20 = None
        volume_ratio = None
        volume_pass = True

        if (
            "volume" in work.columns
            and len(work) >= 20
            and min_vol_mult > 0
        ):
            volume_window = pd.to_numeric(
                work["volume"].iloc[-21:-1],
                errors="coerce",
            )

            if (
                len(volume_window) == 20
                and volume_window.notna().all()
            ):
                volume_sma20 = float(
                    volume_window.mean()
                )

                if (
                    volume_sma20 > 0
                    and current_volume is not None
                ):
                    volume_ratio = (
                        current_volume
                        / volume_sma20
                    )

                    volume_pass = (
                        current_volume
                        >= (
                            volume_sma20
                            * min_vol_mult
                        )
                    )

        result[
            "bars_after_event"
        ] = 1

        result[
            "bars_considered"
        ] = 1

        result[
            "breakout_pass"
        ] = bool(
            breakout_pass
        )

        result[
            "volume_pass"
        ] = bool(
            volume_pass
        )

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

        result[
            "volume_sma20"
        ] = volume_sma20

        result[
            "volume_ratio"
        ] = volume_ratio

        detail = {
            "close_ts": (
                int(h["close_time"])
                if (
                    "close_time"
                    in work.columns
                    and pd.notna(
                        h["close_time"]
                    )
                )
                else None
            ),
            "delay_min": None,
            "previous_high": previous_high,
            "previous_low": previous_low,
            "current_close": current_close,
            "current_volume": current_volume,
            "volume_sma20": volume_sma20,
            "volume_ratio": volume_ratio,
            "breakout_pass": bool(
                breakout_pass
            ),
            "volume_pass": bool(
                volume_pass
            ),
        }

        result[
            "scanned_bar_details"
        ].append(
            detail
        )

        if not breakout_pass:
            result["reason"] = (
                "breakout_failed"
            )
            return result

        if not volume_pass:
            result["reason"] = (
                "volume_failed"
            )
            return result

        result["ok"] = True
        result["reason"] = "passed"

        return result

    # ============================================================
    # EVENT-AWARE MODE
    # ============================================================

    try:
        event_ts = int(
            event_detected_at_ts
        )
    except (
        TypeError,
        ValueError,
    ):
        result["reason"] = (
            "invalid_event_timestamp"
        )
        result["data_pass"] = False
        return result

    work = work.dropna(
        subset=["close_time"]
    )

    work = (
        work.sort_values(
            "close_time"
        )
        .reset_index(drop=True)
    )

    if len(work) < 2:
        result["reason"] = (
            "insufficient_data"
        )
        result["data_pass"] = False
        return result

    candidate_indices: list[int] = []

    for i in range(
        1,
        len(work),
    ):

        close_ts = int(
            work[
                "close_time"
            ].iloc[i]
        )

        if close_ts <= event_ts:
            continue

        delay_min = (
            close_ts - event_ts
        ) / 60000.0

        if delay_min < 0:
            continue

        if (
            delay_min
            > max_trigger_delay_min
        ):
            continue

        candidate_indices.append(i)

    result[
        "bars_after_event"
    ] = len(
        candidate_indices
    )

    if not candidate_indices:
        result["reason"] = (
            "no_trigger_window"
        )
        return result

    saw_breakout = False
    saw_breakout_without_volume = False
    last_diagnostic = None

    for i in candidate_indices:

        h = work.iloc[i]
        p = work.iloc[i - 1]

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

            close_ts = int(
                h["close_time"]
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        current_volume = None

        if (
            "volume" in work.columns
            and pd.notna(
                h["volume"]
            )
        ):
            try:
                current_volume = float(
                    h["volume"]
                )
            except (
                TypeError,
                ValueError,
            ):
                current_volume = None

        delay_min = (
            close_ts - event_ts
        ) / 60000.0

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

        volume_sma20 = None
        volume_ratio = None
        volume_pass = True

        if (
            "volume" in work.columns
            and i >= 20
            and min_vol_mult > 0
        ):

            volume_window = pd.to_numeric(
                work[
                    "volume"
                ].iloc[
                    i - 20:i
                ],
                errors="coerce",
            )

            if (
                len(volume_window)
                == 20
                and volume_window.notna().all()
            ):

                volume_sma20 = float(
                    volume_window.mean()
                )

                if (
                    volume_sma20 > 0
                    and current_volume is not None
                ):

                    volume_ratio = (
                        current_volume
                        / volume_sma20
                    )

                    volume_pass = (
                        current_volume
                        >= (
                            volume_sma20
                            * min_vol_mult
                        )
                    )

        detail = {
            "close_ts": close_ts,
            "delay_min": round(
                delay_min,
                3,
            ),
            "previous_high": previous_high,
            "previous_low": previous_low,
            "current_close": current_close,
            "current_volume": current_volume,
            "volume_sma20": volume_sma20,
            "volume_ratio": volume_ratio,
            "breakout_pass": bool(
                breakout_pass
            ),
            "volume_pass": bool(
                volume_pass
            ),
        }

        result[
            "scanned_bar_details"
        ].append(
            detail
        )

        result[
            "bars_considered"
        ] += 1

        last_diagnostic = detail

        if not breakout_pass:
            continue

        saw_breakout = True

        if not volume_pass:
            saw_breakout_without_volume = True
            continue

        result.update(
            {
                "ok": True,
                "reason": "passed",
                "breakout_pass": True,
                "volume_pass": True,
                "trigger_bar_close_ts": close_ts,
                "trigger_delay_min": round(
                    delay_min,
                    3,
                ),
                "previous_high": previous_high,
                "previous_low": previous_low,
                "current_close": current_close,
                "current_volume": current_volume,
                "volume_sma20": volume_sma20,
                "volume_ratio": volume_ratio,
            }
        )

        return result

    if (
        saw_breakout
        and saw_breakout_without_volume
    ):
        result["reason"] = (
            "volume_failed"
        )

    else:
        result["reason"] = (
            "breakout_failed"
        )

    if last_diagnostic:

        result.update(
            {
                "previous_high": (
                    last_diagnostic[
                        "previous_high"
                    ]
                ),
                "previous_low": (
                    last_diagnostic[
                        "previous_low"
                    ]
                ),
                "current_close": (
                    last_diagnostic[
                        "current_close"
                    ]
                ),
                "current_volume": (
                    last_diagnostic[
                        "current_volume"
                    ]
                ),
                "volume_sma20": (
                    last_diagnostic[
                        "volume_sma20"
                    ]
                ),
                "volume_ratio": (
                    last_diagnostic[
                        "volume_ratio"
                    ]
                ),
                "breakout_pass": bool(
                    saw_breakout
                ),
                "volume_pass": not (
                    saw_breakout_without_volume
                ),
            }
        )

    return result


def build_15m_trigger(
    df15: pd.DataFrame,
    direction: str,
    min_vol_mult: float = 1.0,
    event_detected_at_ts: int | None = None,
    max_trigger_delay_min: float = 60.0,
) -> bool:
    """
    Backward-compatible 15M trigger.

    Старые вызовы:
        build_15m_trigger(df, "LONG")

    Новый production-вызов:
        build_15m_trigger(
            df,
            "LONG",
            event_detected_at_ts=...,
            max_trigger_delay_min=60,
        )
    """

    diagnostic = diagnose_15m_trigger(
        df15=df15,
        direction=direction,
        event_detected_at_ts=event_detected_at_ts,
        max_trigger_delay_min=max_trigger_delay_min,
        min_vol_mult=min_vol_mult,
    )

    return bool(
        diagnostic.get(
            "ok",
            False,
        )
    )


def check_btc_regime(
    btc_1h_df: pd.DataFrame,
    direction: str,
) -> tuple[bool, str]:

    if len(btc_1h_df) < 5:
        return True, "INSUFFICIENT_DATA"

    if "close" not in btc_1h_df.columns:
        return True, "INSUFFICIENT_DATA"

    close = pd.to_numeric(
        btc_1h_df["close"],
        errors="coerce",
    )

    if close.isna().any():
        return True, "INSUFFICIENT_DATA"

    last_close = float(
        close.iloc[-1]
    )

    prev_1h = float(
        close.iloc[-2]
    )

    prev_4h = float(
        close.iloc[-5]
    )

    if (
        last_close <= 0
        or prev_1h <= 0
        or prev_4h <= 0
    ):
        return True, "INSUFFICIENT_DATA"

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
                f"BTC_DUMPING_1H "
                f"({chg_1h_pct:.2f}%)",
            )

        if chg_4h_pct < -2.5:
            return (
                False,
                f"BTC_DUMPING_4H "
                f"({chg_4h_pct:.2f}%)",
            )

    elif d == "SHORT":

        if chg_1h_pct > 1.5:
            return (
                False,
                f"BTC_PUMPING_1H "
                f"(+{chg_1h_pct:.2f}%)",
            )

        if chg_4h_pct > 3.0:
            return (
                False,
                f"BTC_PUMPING_4H "
                f"(+{chg_4h_pct:.2f}%)",
            )

    return True, "OK"
