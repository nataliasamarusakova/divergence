import pandas as pd
import pytest

from event_engine.trend_filter import evaluate_trend_filter


BASE_TS = 1_700_000_000_000
H1 = 3_600_000
H4 = 14_400_000


def _frame(n: int, step: float, interval_ms: int, *, base: float = 100.0, start_ts: int = BASE_TS) -> pd.DataFrame:
    rows = []
    price = base
    for i in range(n):
        close = price + step * i
        rows.append(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1000.0,
                "close_time": start_ts + i * interval_ms,
            }
        )
    return pd.DataFrame(rows)


def _result(df1, df4, direction="LONG", decision_ts=None, **kwargs):
    ts = int(decision_ts if decision_ts is not None else max(df1["close_time"].iloc[-1], df4["close_time"].iloc[-1]))
    return evaluate_trend_filter(
        symbol="TESTUSDT",
        direction=direction,
        event_type="HIDDEN_BULLISH_RSI" if direction == "LONG" else "HIDDEN_BEARISH_RSI",
        df_1h=df1,
        df_4h=df4,
        btc_1h_df=_frame(10, 1.0, H1),
        decision_ts_ms=ts,
        min_bars_1h=400,
        min_bars_4h=400,
        persistence_lookback_1h=6,
        persistence_lookback_4h=3,
        slope_lookback_4h=6,
        **kwargs,
    )


def test_trend_filter_allows_causal_bullish_alignment():
    df1 = _frame(450, 1.0, H1)
    df4 = _frame(450, 2.0, H4)
    out = _result(df1, df4, "LONG")
    assert out["trend_decision"] == "ALIGNED"
    assert out["trend_4h"] == "BULL"
    assert out["trend_1h"] == "LONG"
    assert out["trend_persistence"] == "PERSISTENT"
    assert out["trend_4h_bar_close_ts"] <= out["decision_ts"]
    assert out["trend_1h_bar_close_ts"] <= out["decision_ts"]


def test_trend_filter_rejects_bearish_direction_against_bull_regime():
    df1 = _frame(450, 1.0, H1)
    df4 = _frame(450, 2.0, H4)
    out = _result(df1, df4, "SHORT")
    assert out["trend_decision"] == "REJECT"
    assert out["trend_reject_reason"] == "TREND_4H_DIRECTION_MISMATCH"
    assert out["trend_4h"] == "BULL"


def test_trend_filter_reason_priority_uses_4h_before_1h():
    df1 = _frame(450, -1.0, H1)
    df4 = _frame(450, 0.0, H4)
    out = _result(df1, df4, "LONG")
    assert out["trend_reject_reason"] == "TREND_4H_TRANSITION"


def test_trend_filter_requires_warm_4h_history():
    df1 = _frame(450, 1.0, H1)
    df4 = _frame(300, 2.0, H4)
    out = _result(df1, df4, "LONG")
    assert out["trend_reject_reason"] == "TREND_4H_UNKNOWN"
    assert out["trend_decision"] == "REJECT"


def test_trend_filter_requires_warm_1h_history_after_4h_alignment():
    df1 = _frame(300, 1.0, H1)
    df4 = _frame(450, 2.0, H4)
    out = _result(df1, df4, "LONG")
    assert out["trend_reject_reason"] == "TREND_1H_UNKNOWN"
    assert out["trend_decision"] == "REJECT"


def test_trend_filter_is_causal_and_ignores_future_bars():
    df1_before = _frame(450, 1.0, H1)
    df4_before = _frame(450, 2.0, H4)
    decision_ts = max(df1_before["close_time"].iloc[-1], df4_before["close_time"].iloc[-1])

    future_1h = _frame(40, -8.0, H1, base=float(df1_before["close"].iloc[-1]), start_ts=int(decision_ts + H1))
    future_4h = _frame(40, -16.0, H4, base=float(df4_before["close"].iloc[-1]), start_ts=int(decision_ts + H4))
    df1 = pd.concat([df1_before, future_1h], ignore_index=True)
    df4 = pd.concat([df4_before, future_4h], ignore_index=True)

    out = _result(df1, df4, "LONG", decision_ts=decision_ts)
    assert out["trend_decision"] == "ALIGNED"
    assert out["trend_4h"] == "BULL"
    assert out["trend_1h"] == "LONG"
    assert out["trend_4h_bar_close_ts"] == int(df4_before["close_time"].iloc[-1])
    assert out["trend_1h_bar_close_ts"] == int(df1_before["close_time"].iloc[-1])


def test_trend_filter_self_btc_does_not_apply_cross_asset_btc_veto():
    df1 = _frame(450, 1.0, H1)
    df4 = _frame(450, 2.0, H4)
    out = evaluate_trend_filter(
        symbol="BTCUSDT",
        direction="LONG",
        event_type="BREAKOUT",
        df_1h=df1,
        df_4h=df4,
        btc_1h_df=_frame(10, -1.0, H1),
        decision_ts_ms=int(df4["close_time"].iloc[-1]),
        min_bars_1h=400,
        min_bars_4h=400,
    )
    assert out["trend_decision"] == "ALIGNED"
    assert out["btc_regime"] == "SELF"
    assert out["btc_context_veto"] is False


def test_trend_filter_persistence_can_be_required_without_changing_direction_logic():
    df4 = _frame(450, 2.0, H4)
    df1 = _frame(444, 1.0, H1)
    # Keep the 1H structural direction bullish but make the most recent
    # persistence window negative enough to fail the optional requirement.
    last = len(df1) - 1
    for idx, value in zip(range(last - 5, last + 1), [550.0, 545.0, 540.0, 537.0, 535.0, 533.0]):
        df1.loc[idx, "close"] = value
    df1["open"] = df1["close"]
    df1["high"] = df1["close"] + 0.5
    df1["low"] = df1["close"] - 0.5
    out = _result(df1, df4, "LONG", require_persistence=True)
    assert out["trend_4h"] == "BULL"
    assert out["trend_1h"] in {"LONG", "TRANSITION"}
    assert out["trend_reject_reason"] in {"TREND_PERSISTENCE_TRANSITION", "TREND_1H_TRANSITION"}


def test_trend_filter_enforce_rejects_unknown_snapshot_but_shadow_does_not():
    import run_once
    snapshot = {"trend_decision": "REJECT", "trend_reject_reason": "TREND_4H_UNKNOWN"}
    assert run_once._trend_filter_enforce_reject(snapshot, "shadow") is False
    assert run_once._trend_filter_enforce_reject(snapshot, "enforce") is True
    assert run_once._trend_filter_enforce_reject({"trend_decision": "ALIGNED"}, "enforce") is False
    assert run_once._trend_filter_enforce_reject(None, "enforce") is True



def test_trend_filter_4h_slope_lookback_is_independent_from_persistence():
    df1 = _frame(450, 1.0, H1)
    df4 = _frame(450, 2.0, H4)
    out = _result(df1, df4, "LONG")
    assert out["slope_lookback_4h"] == 6
    assert out["trend_4h_slope_lookback"] == 6
    assert out["persistence_lookback_4h"] == 3
    ema50 = df4["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    expected_slope = (float(ema50.iloc[-1]) / float(ema50.iloc[-7]) - 1.0) * 100.0
    assert out["trend_4h_ema50_slope_pct"] == pytest.approx(expected_slope)


def test_trend_filter_can_use_400_plus_buffer_when_trigger_precedes_latest_htf_close(monkeypatch):
    import run_once

    df = _frame(400, 1.0, H1)
    decision_ts = int(df["close_time"].iloc[-1] - 15 * 60 * 1000)
    calls = []

    def fake_fetch(symbol, timeframe, limit):
        calls.append((symbol, timeframe, limit))
        interval = H1 if timeframe == "1h" else H4
        expanded = _frame(412, 1.0, interval, start_ts=BASE_TS - 12 * interval)
        return expanded.to_dict("records")

    monkeypatch.setattr(run_once, "_fetch_market_klines_scan", fake_fetch)
    out, refetched = run_once._ensure_trend_history(
        symbol="TESTUSDT", timeframe="1h", frame=df, decision_ts=decision_ts, min_bars=400
    )

    assert refetched is True
    assert len(out) == 412
    assert calls == [("TESTUSDT", "1h", 412)]
    assert run_once._trend_history_causal_count(out, decision_ts) >= 400


def test_trend_history_does_not_refetch_when_causal_bars_are_sufficient(monkeypatch):
    import run_once
    df = _frame(412, 1.0, H1)
    decision_ts = int(df["close_time"].iloc[-1])
    calls = []
    monkeypatch.setattr(run_once, "_fetch_market_klines_scan", lambda *args: calls.append(args) or [])
    out, refetched = run_once._ensure_trend_history(
        symbol="TESTUSDT", timeframe="1h", frame=df, decision_ts=decision_ts, min_bars=400
    )
    assert refetched is False
    assert out is df
    assert calls == []
