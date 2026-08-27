def diagnose_15m_trigger(
    df15: pd.DataFrame,
    direction: str,
    event_detected_at_ts: int | None = None,
    max_trigger_delay_min: float = 60.0,
    min_vol_mult: float = 1.05,
) -> dict[str, Any]:
    """
    Диагностика 15M trigger.

    Два режима:

    1. event_detected_at_ts is None:
       backward-compatible режим для старых вызовов/тестов.
       Проверяется последняя свеча относительно предыдущей.

    2. event_detected_at_ts задан:
       event-aware режим.
       Проверяются только свечи, закрывшиеся после события
       и не позднее max_trigger_delay_min.
    """

    result: dict[str, Any] = {
        "ok": False,
        "reason": None,
        "direction": str(direction).upper(),
        "event_detected_at_ts": event_detected_at_ts,
        "max_trigger_delay_min": float(max_trigger_delay_min),
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

    if not isinstance(df15, pd.DataFrame):
        result["reason"] = "invalid_15m_data"
        result["data_pass"] = False
        return result

    if len(df15) < 2:
        result["reason"] = "insufficient_data"
        result["data_pass"] = False
        return result

    d = str(direction).upper()

    if d not in {"LONG", "SHORT"}:
        result["reason"] = "invalid_direction"
        result["direction_pass"] = False
        return result

    required = {"close", "high", "low"}
    missing = required - set(df15.columns)

    # close_time нужен только для event-aware режима.
    if event_detected_at_ts is not None:
        required_with_time = {
            "close",
            "high",
            "low",
            "close_time",
        }
        missing = required_with_time - set(
            df15.columns
        )

    if missing:
        result["reason"] = "invalid_15m_data"
        result["data_pass"] = False
        result["error"] = (
            "missing columns: "
            + ", ".join(sorted(missing))
        )
        return result

    work = df15.copy()

    for col in ("close", "high", "low"):
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
    ).reset_index(drop=True)

    if len(work) < 2:
        result["reason"] = "insufficient_data"
        result["data_pass"] = False
        return result

    # ============================================================
    # LEGACY / COMPATIBILITY MODE
    # ============================================================

    if event_detected_at_ts is None:

        i = len(work) - 1

        h = work.iloc[i]
        p = work.iloc[i - 1]

        try:
            current_close = float(h["close"])
            previous_high = float(p["high"])
            previous_low = float(p["low"])
        except (TypeError, ValueError):
            result["reason"] = "invalid_15m_data"
            result["data_pass"] = False
            return result

        current_volume = None
        if (
            "volume" in work.columns
            and pd.notna(h["volume"])
        ):
            try:
                current_volume = float(h["volume"])
            except (TypeError, ValueError):
                current_volume = None

        if d == "LONG":
            breakout_pass = (
                current_close > previous_high
            )
        else:
            breakout_pass = (
                current_close < previous_low
            )

        volume_sma20 = None
        volume_ratio = None
        volume_pass = True

        # Для старого теста len=2 -> volume-фильтр не активируется,
        # как и раньше.
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

        result["bars_after_event"] = 1
        result["bars_considered"] = 1
        result["breakout_pass"] = bool(
            breakout_pass
        )
        result["volume_pass"] = bool(
            volume_pass
        )
        result["previous_high"] = previous_high
        result["previous_low"] = previous_low
        result["current_close"] = current_close
        result["current_volume"] = current_volume
        result["volume_sma20"] = volume_sma20
        result["volume_ratio"] = volume_ratio

        detail = {
            "close_ts": (
                int(h["close_time"])
                if (
                    "close_time" in work.columns
                    and pd.notna(h["close_time"])
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

        result["scanned_bar_details"].append(
            detail
        )

        if not breakout_pass:
            result["reason"] = "breakout_failed"
            return result

        if not volume_pass:
            result["reason"] = "volume_failed"
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
    except (TypeError, ValueError):
        result["reason"] = "invalid_event_timestamp"
        result["data_pass"] = False
        return result

    work = work.dropna(
        subset=["close_time"]
    ).sort_values(
        "close_time"
    ).reset_index(drop=True)

    if len(work) < 2:
        result["reason"] = "insufficient_data"
        result["data_pass"] = False
        return result

    candidate_indices: list[int] = []

    for i in range(1, len(work)):

        close_ts = int(
            work["close_time"].iloc[i]
        )

        if close_ts <= event_ts:
            continue

        delay_min = (
            close_ts - event_ts
        ) / 60000.0

        if delay_min < 0:
            continue

        if delay_min > max_trigger_delay_min:
            continue

        candidate_indices.append(i)

    result["bars_after_event"] = len(
        candidate_indices
    )

    if not candidate_indices:
        result["reason"] = "no_trigger_window"
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
        except (TypeError, ValueError):
            continue

        current_volume = None

        if (
            "volume" in work.columns
            and pd.notna(h["volume"])
        ):
            try:
                current_volume = float(
                    h["volume"]
                )
            except (TypeError, ValueError):
                current_volume = None

        delay_min = (
            close_ts - event_ts
        ) / 60000.0

        if d == "LONG":
            breakout_pass = (
                current_close > previous_high
            )
        else:
            breakout_pass = (
                current_close < previous_low
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
                work["volume"].iloc[i - 20:i],
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

        result["scanned_bar_details"].append(
            detail
        )

        result["bars_considered"] += 1
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
        result["reason"] = "volume_failed"
    else:
        result["reason"] = "breakout_failed"

    if last_diagnostic:
        result.update(
            {
                "previous_high": last_diagnostic[
                    "previous_high"
                ],
                "previous_low": last_diagnostic[
                    "previous_low"
                ],
                "current_close": last_diagnostic[
                    "current_close"
                ],
                "current_volume": last_diagnostic[
                    "current_volume"
                ],
                "volume_sma20": last_diagnostic[
                    "volume_sma20"
                ],
                "volume_ratio": last_diagnostic[
                    "volume_ratio"
                ],
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
    Совместимый wrapper.

    Без event_detected_at_ts:
        старое поведение — последняя свеча против предыдущей.

    С event_detected_at_ts:
        новый event-aware trigger.
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
