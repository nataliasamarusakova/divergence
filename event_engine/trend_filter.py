"""Causal, deterministic trend/regime diagnostics for entry candidates.

This module is intentionally isolated from the existing event detectors.  It does
not generate signals, alter Demand/Supply construction, or change the 15M trigger.
The first release is intended to run in shadow mode before becoming an enforce gate.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _clean_closed_frame(df: pd.DataFrame | None, decision_ts_ms: int | float | None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if "close_time" not in work.columns:
        return pd.DataFrame()
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work["close_time"] = pd.to_numeric(work["close_time"], errors="coerce")
    work = work.dropna(subset=["close", "close_time"]).sort_values("close_time").drop_duplicates("close_time", keep="last")
    work = work[work["close"] > 0]
    if decision_ts_ms is not None:
        try:
            decision_ts = int(decision_ts_ms)
        except (TypeError, ValueError):
            decision_ts = 0
        if decision_ts > 0:
            work = work[work["close_time"] <= decision_ts]
    return work.reset_index(drop=True)


def _safe_pct_change(current: float, previous: float) -> float | None:
    if not all(math.isfinite(x) for x in (current, previous)) or previous <= 0:
        return None
    return (current - previous) / previous * 100.0


def _snapshot_frame(df: pd.DataFrame, *, ema_periods: tuple[int, ...], slope_lookback: int, return_lookback: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "bars": int(len(df)),
        "last_bar_close_ts": None,
        "close": None,
        "ema20": None,
        "ema50": None,
        "ema200": None,
        "ema50_slope_pct": None,
        "return_pct": None,
    }
    if df.empty:
        return out
    close = pd.to_numeric(df["close"], errors="coerce")
    ema = {}
    for period in ema_periods:
        ema[period] = close.ewm(span=int(period), adjust=False, min_periods=int(period)).mean()
    out["last_bar_close_ts"] = int(df["close_time"].iloc[-1])
    out["close"] = float(close.iloc[-1])
    for period in (20, 50, 200):
        series = ema.get(period)
        value = series.iloc[-1] if series is not None else float("nan")
        out[f"ema{period}"] = float(value) if pd.notna(value) else None
    slope_idx = len(df) - 1 - int(slope_lookback)
    ema50 = ema[50]
    if slope_idx >= 0 and pd.notna(ema50.iloc[-1]) and pd.notna(ema50.iloc[slope_idx]) and float(ema50.iloc[slope_idx]) > 0:
        out["ema50_slope_pct"] = float((float(ema50.iloc[-1]) / float(ema50.iloc[slope_idx]) - 1.0) * 100.0)
    return_idx = len(df) - 1 - int(return_lookback)
    if return_idx >= 0:
        out["return_pct"] = _safe_pct_change(float(close.iloc[-1]), float(close.iloc[return_idx]))
    return out


def _classify_btc(df: pd.DataFrame, decision_ts_ms: int | float | None) -> dict[str, Any]:
    work = _clean_closed_frame(df, decision_ts_ms)
    out: dict[str, Any] = {
        "state": "UNKNOWN",
        "bars": int(len(work)),
        "bar_close_ts": int(work["close_time"].iloc[-1]) if not work.empty else None,
        "chg_1h_pct": None,
        "chg_4h_pct": None,
    }
    if len(work) < 5:
        return out
    close = pd.to_numeric(work["close"], errors="coerce")
    last = float(close.iloc[-1])
    prev_1h = float(close.iloc[-2])
    prev_4h = float(close.iloc[-5])
    out["chg_1h_pct"] = _safe_pct_change(last, prev_1h)
    out["chg_4h_pct"] = _safe_pct_change(last, prev_4h)
    if out["chg_1h_pct"] is None or out["chg_4h_pct"] is None:
        return out
    c1 = float(out["chg_1h_pct"])
    c4 = float(out["chg_4h_pct"])
    if c1 > 0 and c4 > 0:
        out["state"] = "BULLISH"
    elif c1 < 0 and c4 < 0:
        out["state"] = "BEARISH"
    else:
        out["state"] = "MIXED"
    return out


def evaluate_trend_filter(
    *,
    symbol: str,
    direction: str,
    event_type: str | None,
    df_1h: pd.DataFrame | None,
    df_4h: pd.DataFrame | None,
    btc_1h_df: pd.DataFrame | None,
    decision_ts_ms: int | float,
    min_bars_1h: int = 400,
    min_bars_4h: int = 400,
    persistence_lookback_1h: int = 6,
    persistence_lookback_4h: int = 3,
    slope_lookback_4h: int = 6,
    require_persistence: bool = False,
    mode: str = "shadow",
) -> dict[str, Any]:
    """Evaluate a candidate against causal 4H regime + 1H direction.

    All available diagnostics are computed before the single prioritized reject
    reason is selected. The returned ``decision`` is diagnostic unless the caller
    explicitly enforces it. Missing/transition data is never converted into a
    positive trend decision. BTC is contextual only in v1; it is intentionally not
    a hard veto here.
    """
    d = str(direction or "").upper()
    symbol_u = str(symbol or "").upper()
    event_u = str(event_type or "").upper()
    decision_ts = int(decision_ts_ms or 0)
    mode_norm = str(mode or "shadow").strip().lower()
    if mode_norm not in {"shadow", "enforce"}:
        mode_norm = "shadow"

    base: dict[str, Any] = {
        "version": "trend-v1-2026-10-04",
        "enabled": True,
        "mode": mode_norm,
        "symbol": symbol_u,
        "event_direction": d,
        "event_type": event_u or None,
        "decision_ts": decision_ts,
        "trend_decision": "REJECT",
        "trend_reject_reason": None,
        "trend_4h": "UNKNOWN",
        "trend_1h": "UNKNOWN",
        "trend_persistence": "UNKNOWN",
        "trend_4h_bars": 0,
        "trend_1h_bars": 0,
        "trend_4h_bar_close_ts": None,
        "trend_1h_bar_close_ts": None,
        "trend_persistence_1h_return_pct": None,
        "trend_persistence_4h_return_pct": None,
        "btc_regime": "SELF" if symbol_u in {"BTC", "BTCUSDT", "BTC-USDT"} else "UNKNOWN",
        "btc_bar_close_ts": None,
        "btc_chg_1h_pct": None,
        "btc_chg_4h_pct": None,
        "btc_context_veto": False,
        "require_persistence": bool(require_persistence),
        "min_bars_1h": int(min_bars_1h),
        "min_bars_4h": int(min_bars_4h),
        "persistence_lookback_1h": int(persistence_lookback_1h),
        "persistence_lookback_4h": int(persistence_lookback_4h),
        "slope_lookback_4h": int(slope_lookback_4h),
    }

    btc = _classify_btc(btc_1h_df, decision_ts)
    base.update({
        "btc_regime": "SELF" if symbol_u in {"BTC", "BTCUSDT", "BTC-USDT"} else btc["state"],
        "btc_bar_close_ts": btc["bar_close_ts"],
        "btc_chg_1h_pct": btc["chg_1h_pct"],
        "btc_chg_4h_pct": btc["chg_4h_pct"],
        "btc_bars": btc["bars"],
    })

    if d not in {"LONG", "SHORT"}:
        base["trend_reject_reason"] = "TREND_EVENT_DIRECTION_UNKNOWN"
        return base
    if decision_ts <= 0:
        base["trend_reject_reason"] = "TREND_DECISION_TS_INVALID"
        return base

    h1 = _clean_closed_frame(df_1h, decision_ts)
    h4 = _clean_closed_frame(df_4h, decision_ts)
    base["trend_1h_bars"] = int(len(h1))
    base["trend_4h_bars"] = int(len(h4))

    s4: dict[str, Any] | None = None
    s1: dict[str, Any] | None = None
    if len(h4) >= int(min_bars_4h):
        s4 = _snapshot_frame(
            h4,
            ema_periods=(20, 50, 200),
            slope_lookback=max(1, slope_lookback_4h),
            return_lookback=max(1, persistence_lookback_4h),
        )
        base.update({
            "trend_4h_bar_close_ts": s4["last_bar_close_ts"],
            "trend_4h_close": s4["close"],
            "trend_4h_ema50": s4["ema50"],
            "trend_4h_ema200": s4["ema200"],
            "trend_4h_ema50_slope_pct": s4["ema50_slope_pct"],
            "trend_persistence_4h_return_pct": s4["return_pct"],
            "trend_4h_slope_lookback": max(1, int(slope_lookback_4h)),
        })
        if None not in (s4["close"], s4["ema50"], s4["ema200"], s4["ema50_slope_pct"]):
            if float(s4["close"]) > float(s4["ema200"]) and float(s4["ema50"]) > float(s4["ema200"]) and float(s4["ema50_slope_pct"]) > 0:
                base["trend_4h"] = "BULL"
            elif float(s4["close"]) < float(s4["ema200"]) and float(s4["ema50"]) < float(s4["ema200"]) and float(s4["ema50_slope_pct"]) < 0:
                base["trend_4h"] = "BEAR"
            else:
                base["trend_4h"] = "TRANSITION"

    if len(h1) >= int(min_bars_1h):
        s1 = _snapshot_frame(
            h1,
            ema_periods=(20, 50, 200),
            slope_lookback=max(1, persistence_lookback_1h),
            return_lookback=max(1, persistence_lookback_1h),
        )
        base.update({
            "trend_1h_bar_close_ts": s1["last_bar_close_ts"],
            "trend_1h_close": s1["close"],
            "trend_1h_ema20": s1["ema20"],
            "trend_1h_ema50": s1["ema50"],
            "trend_1h_ema200": s1["ema200"],
            "trend_1h_ema50_slope_pct": s1["ema50_slope_pct"],
            "trend_persistence_1h_return_pct": s1["return_pct"],
        })
        if None not in (s1["close"], s1["ema20"], s1["ema50"], s1["ema50_slope_pct"]):
            if float(s1["close"]) > float(s1["ema50"]) and float(s1["ema20"]) > float(s1["ema50"]) and float(s1["ema50_slope_pct"]) > 0:
                base["trend_1h"] = "LONG"
            elif float(s1["close"]) < float(s1["ema50"]) and float(s1["ema20"]) < float(s1["ema50"]) and float(s1["ema50_slope_pct"]) < 0:
                base["trend_1h"] = "SHORT"
            else:
                base["trend_1h"] = "TRANSITION"

    ret1 = s1["return_pct"] if s1 is not None else None
    ret4 = s4["return_pct"] if s4 is not None else None
    persistence_ok = (
        ret1 is not None and ret4 is not None and
        ((d == "LONG" and ret1 > 0 and ret4 > 0) or (d == "SHORT" and ret1 < 0 and ret4 < 0))
    )
    if ret1 is not None or ret4 is not None:
        base["trend_persistence"] = "PERSISTENT" if persistence_ok else "TRANSITION"

    # Fixed priority: data → 4H state → 1H state → optional persistence.
    if len(h4) < int(min_bars_4h) or s4 is None or None in (s4["close"], s4["ema50"], s4["ema200"], s4["ema50_slope_pct"]):
        base["trend_reject_reason"] = "TREND_4H_UNKNOWN"
        return base
    expected_4h = "BULL" if d == "LONG" else "BEAR"
    if base["trend_4h"] == "TRANSITION":
        base["trend_reject_reason"] = "TREND_4H_TRANSITION"
        return base
    if base["trend_4h"] != expected_4h:
        base["trend_reject_reason"] = "TREND_4H_DIRECTION_MISMATCH"
        return base

    if len(h1) < int(min_bars_1h) or s1 is None or None in (s1["close"], s1["ema20"], s1["ema50"], s1["ema50_slope_pct"]):
        base["trend_reject_reason"] = "TREND_1H_UNKNOWN"
        return base
    if base["trend_1h"] == "TRANSITION":
        base["trend_reject_reason"] = "TREND_1H_TRANSITION"
        return base
    if base["trend_1h"] != d:
        base["trend_reject_reason"] = "TREND_1H_DIRECTION_MISMATCH"
        return base

    if require_persistence and not persistence_ok:
        base["trend_reject_reason"] = "TREND_PERSISTENCE_TRANSITION"
        return base

    base["trend_decision"] = "ALIGNED"
    base["trend_reject_reason"] = None
    return base

