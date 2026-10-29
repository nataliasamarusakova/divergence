from __future__ import annotations

import pytest

from pine_r541_sr import (
    Candle,
    _is_pivot_high,
    _is_pivot_low,
    build_confirmed_pivots,
    calculate_event_sr,
    compute_current_sr,
)


def _candle(ts: int, high: float, low: float) -> Candle:
    close = (high + low) / 2.0
    return Candle(ts=ts, open=close, high=high, low=low, close=close, volume=100.0)


def test_pivot_helpers_reject_edges_and_find_local_extrema():
    candles = [
        _candle(1, 2.0, 1.0),
        _candle(2, 4.0, 2.0),
        _candle(3, 9.0, 3.0),
        _candle(4, 5.0, 2.0),
        _candle(5, 3.0, 1.0),
    ]
    assert _is_pivot_high(candles, 0, 1) is False
    assert _is_pivot_low(candles, 4, 1) is False
    assert _is_pivot_high(candles, 2, 1) is True
    assert _is_pivot_low(candles, 2, 1) is False
    assert _is_pivot_low(candles, 4, 1) is False  # right neighbour is out of range


def test_build_confirmed_pivots_emits_on_confirmation_bar_not_pivot_bar():
    candles = [
        _candle(1, 2.0, 1.0),
        _candle(2, 4.0, 2.0),
        _candle(3, 9.0, 3.0),
        _candle(4, 5.0, 2.0),
        _candle(5, 3.0, 1.0),
        _candle(6, 2.5, 1.5),
    ]
    ph, pl = build_confirmed_pivots(candles, rb=1)
    assert ph[2] is None
    assert ph[3] == 9.0
    assert pl[5] == 1.0
    assert all(value is None for value in pl[:5])


def test_event_sr_clusters_nearby_pivots_using_channel_width():
    candles = [
        _candle(1, 12.0, 8.0),
        _candle(2, 11.5, 8.5),
        _candle(3, 11.3, 8.6),
        _candle(4, 11.0, 8.4),
    ]
    ph = [None, None, 11.2, 11.0]
    pl = [None, None, None, None]
    out = calculate_event_sr(candles, ph, pl, event_idx=3, rb=1, prd=3, channel_w=10.0, strength_sr=2)
    assert out["cwidth"] == pytest.approx(0.31)
    assert out["levels"] == [11.0]
    assert out["accepted"] == [
        {"countpp": 1, "price": 11.0, "kind": "H", "cluster_width": pytest.approx(0.31), "points": 2}
    ]


def test_event_sr_preserves_documented_pivot_high_then_low_overwrite_order():
    candles = [
        _candle(1, 12.0, 8.0),
        _candle(2, 11.5, 8.5),
        _candle(3, 11.3, 8.6),
        _candle(4, 11.0, 8.4),
    ]
    ph = [None, None, 11.2, 11.0]
    pl = [None, None, None, 9.0]
    out = calculate_event_sr(candles, ph, pl, event_idx=3, rb=1, prd=3, channel_w=10.0, strength_sr=2)
    # The implementation documents Pine assignment order: ph is stored first,
    # then pl overwrites the same sr_levels slot when both are present.
    assert out["levels"] == [9.0]
    assert out["accepted"][0]["kind"] == "L"
    assert out["accepted"][0]["price"] == 9.0


def test_event_sr_without_pivots_returns_window_bounds_and_no_levels():
    candles = [_candle(1, 12.0, 8.0), _candle(2, 11.0, 9.0), _candle(3, 10.0, 9.5)]
    out = calculate_event_sr(candles, [None] * 3, [None] * 3, event_idx=2, prd=3)
    assert out["levels"] == []
    assert out["accepted"] == []
    assert out["highestph"] == 8.0
    assert out["lowestpl"] == 12.0
    assert out["cwidth"] == pytest.approx(0.4)


def test_compute_current_sr_requires_minimum_closed_candle_count():
    candles = [_candle(i, 2.0 + i, 1.0 + i) for i in range(5)]
    with pytest.raises(ValueError, match=r"Need at least 6 candles, got 5"):
        compute_current_sr(candles, rb=1, prd=3)


def test_compute_current_sr_empty_fallback_is_explicit_when_no_pivot_exists():
    candles = [_candle(i, 10.0 + i, 1.0 + i) for i in range(6)]
    out = compute_current_sr(candles, rb=1, prd=3)
    assert out["levels"] == []
    assert out["sr_levels"] == [None] * 21
    assert out["highestph"] is None
    assert out["lowestpl"] is None
    assert out["cwidth"] is None
    assert out["event_idx"] is None
    assert out["latest_closed_timestamp"] == candles[-1].ts
    assert out["latest_closed_close"] == candles[-1].close


def test_compute_current_sr_returns_latest_pivot_event_and_latest_close_metadata():
    candles = [
        _candle(1, 2.0, 1.0),
        _candle(2, 4.0, 2.0),
        _candle(3, 9.0, 3.0),
        _candle(4, 5.0, 2.0),
        _candle(5, 3.0, 1.0),
        _candle(6, 2.5, 1.5),
        _candle(7, 2.4, 1.6),
        _candle(8, 2.3, 1.7),
    ]
    out = compute_current_sr(candles, rb=1, prd=3)
    assert out["event_idx"] == 5
    assert out["event_timestamp"] == candles[5].ts
    assert out["latest_closed_timestamp"] == candles[-1].ts
    assert out["latest_closed_close"] == candles[-1].close
    assert out["pivot_high_confirmed_at_latest"] is None
    assert out["pivot_low_confirmed_at_latest"] is None
