from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_open_orders_http_429_persists_cooldown_for_signed_nonretryable_get(tmp_path, monkeypatch):
    from event_engine import bingx as bx

    class Response:
        status_code = 429
        headers = {"Content-Type": "application/json", "Retry-After": "30"}

        def json(self):
            return {"code": 0, "msg": "Too Many Requests", "data": []}

    class Session:
        def __init__(self):
            self.calls = 0

        def request(self, **kwargs):
            self.calls += 1
            return Response()

    session = Session()
    cooldown_path = tmp_path / "open_orders_cooldown.json"
    monkeypatch.setattr(bx, "API_KEY", "test-key")
    monkeypatch.setattr(bx, "SECRET_KEY", "test-secret")
    monkeypatch.setattr(bx, "FAST_SESSION", session)
    monkeypatch.setattr(bx, "OPEN_ORDERS_COOLDOWN_STATE_PATH", cooldown_path)
    monkeypatch.setattr(bx, "_apply_request_timestamp", lambda params: None)
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_UNTIL_MS", 0)
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_REASON", "")

    diagnostics = {}
    result = bx.get_open_orders_snapshot(retryable=False, diagnostics=diagnostics)

    assert result["status"] == "error"
    assert result["rate_limited"] is True
    assert result["http_status"] == 429
    assert diagnostics["rate_limited"] is True
    assert session.calls == 1
    state = json.loads(cooldown_path.read_text(encoding="utf-8"))
    assert state["cooldown_until_ms"] > int(time.time() * 1000)

    # The persisted cooldown must stop the next call before it reaches HTTP.
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_UNTIL_MS", 0)
    second = bx.get_open_orders_snapshot(retryable=False)
    assert second["rate_limited"] is True
    assert second["local_cooldown"] is True
    assert session.calls == 1


def test_open_orders_html_429_cooldown_does_not_depend_on_retryable_flag(tmp_path, monkeypatch):
    from event_engine import bingx as bx

    class Response:
        status_code = 429
        headers = {"Content-Type": "text/html"}

        def json(self):
            raise ValueError("HTML gateway response")

    class Session:
        def __init__(self):
            self.calls = 0

        def request(self, **kwargs):
            self.calls += 1
            return Response()

    session = Session()
    monkeypatch.setattr(bx, "API_KEY", "test-key")
    monkeypatch.setattr(bx, "SECRET_KEY", "test-secret")
    monkeypatch.setattr(bx, "FAST_SESSION", session)
    monkeypatch.setattr(bx, "OPEN_ORDERS_COOLDOWN_STATE_PATH", tmp_path / "cooldown.json")
    monkeypatch.setattr(bx, "_apply_request_timestamp", lambda params: None)
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_UNTIL_MS", 0)
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_REASON", "")

    result = bx._request("GET", bx.OPEN_ORDERS_PATH, {}, signed=True, retryable=False)
    assert result["rate_limited"] is True
    assert result["local_cooldown"] is True
    assert session.calls == 1


def _trigger_frame(*, n=21, include_volume=True):
    data = {
        "high": [100.0] * n,
        "low": [95.0] * n,
        "close": [99.0] * (n - 1) + [101.0],
    }
    if include_volume:
        data["volume"] = [100.0] * (n - 1) + [110.0]
    return pd.DataFrame(data)


def test_15m_trigger_volume_confirmation_fails_closed_for_missing_or_invalid_data():
    from event_engine.signals import diagnose_15m_trigger

    passing = diagnose_15m_trigger(_trigger_frame(), "LONG", require_event_ts=False)
    assert passing["ok"] is True
    assert passing["volume_pass"] is True

    cases = [
        _trigger_frame(n=20),
        _trigger_frame(include_volume=False),
    ]
    bad_history = _trigger_frame()
    bad_history.loc[0, "volume"] = np.nan
    cases.append(bad_history)
    bad_current = _trigger_frame()
    bad_current.loc[20, "volume"] = np.nan
    cases.append(bad_current)

    for frame in cases:
        result = diagnose_15m_trigger(frame, "LONG", require_event_ts=False)
        assert result["ok"] is False
        assert result["volume_pass"] is False
        assert result["reason"] == "volume_failed"


def test_15m_trigger_event_timestamp_branch_uses_same_fail_closed_volume_gate():
    from event_engine.signals import diagnose_15m_trigger

    frame = _trigger_frame()
    frame["close_time"] = np.arange(len(frame), dtype=np.int64) * 900_000 + 899_999
    event_ts = int(frame.iloc[-2]["close_time"])
    valid = diagnose_15m_trigger(frame, "LONG", event_detected_at_ts=event_ts)
    assert valid["ok"] is True

    frame.loc[0, "volume"] = np.nan
    invalid = diagnose_15m_trigger(frame, "LONG", event_detected_at_ts=event_ts)
    assert invalid["ok"] is False
    assert invalid["reason"] == "volume_failed"


def test_retest_drift_policy_is_reused_for_post_fill_measurement():
    import run_once as ro

    metric = ro._entry_drift_metric_for_policy("RETEST")
    assert metric is ro._entry_adverse_drift_pct
    # A long filled below its signal is a favorable retest, not 2.5% signal decay.
    assert metric(100.0, 97.5, "LONG") == pytest.approx(0.0)

    standard = ro._entry_drift_metric_for_policy("CONTINUATION")
    assert standard is ro._entry_drift_pct
    assert standard(100.0, 97.5, "LONG") == pytest.approx(2.5)


def test_binance_klines_reject_malformed_rows_sort_and_deduplicate(monkeypatch):
    from event_engine.binance import BinanceMarketClient

    client = BinanceMarketClient()
    monkeypatch.setattr(client, "resolve_symbol", lambda symbol: "BTCUSDT")
    now_ms = int(time.time() * 1000)
    base = (now_ms // 60_000 - 10) * 60_000
    monkeypatch.setattr("event_engine.binance.time.time", lambda: now_ms / 1000.0)

    def row(open_time, *, close_time=None, high="110", low="90", volume="10", quote="1000", taker_quote="500"):
        return [open_time, "100", high, low, "105", volume,
                open_time + 59_999 if close_time is None else close_time,
                quote, "5", "5", taker_quote, "reserved"]

    first = row(base)
    second = row(base + 60_000)
    duplicate = row(base, high="111")
    zero_close = row(base + 120_000, close_time=0)
    invalid_geometry = row(base + 180_000, high="99", low="101")
    negative_volume = row(base + 240_000, volume="-1")
    invalid_taker = row(base + 300_000, taker_quote="1002")
    monkeypatch.setattr(client, "_request_json", lambda *_args, **_kwargs: [second, first, duplicate, zero_close, invalid_geometry, negative_volume, invalid_taker])

    bars = client.fetch_klines("BTCUSDT", "1m", 10)
    assert [bar["open_time"] for bar in bars] == [base, base + 60_000, base + 300_000]
    assert len({bar["close_time"] for bar in bars}) == len(bars)
    assert bars[-1]["taker_flow_valid"] is False
    assert bars[-1]["bar_delta_usdt"] is None


def test_bingx_and_binance_resolve_lit_and_trumpsol_aliases(monkeypatch):
    from event_engine import bingx
    from event_engine.binance import BinanceMarketClient

    catalog = {
        "LIGHTER-USDT": {"symbol": "LIGHTER-USDT", "displayName": "LIGHTER-USDT"},
        "TRUMP-USDT": {"symbol": "TRUMP-USDT", "displayName": "TRUMP-USDT"},
    }
    monkeypatch.setattr(bingx, "contracts", lambda **kwargs: catalog)
    assert bingx.get_contract("LIT") is catalog["LIGHTER-USDT"]
    assert bingx.get_contract("TRUMPSOL") is catalog["TRUMP-USDT"]

    client = BinanceMarketClient()
    symbols = {"LIGHTERUSDT": {"symbol": "LIGHTERUSDT"}, "TRUMPUSDT": {"symbol": "TRUMPUSDT"}}
    monkeypatch.setattr(client, "refresh_exchange_info", lambda **kwargs: symbols)
    assert client.resolve_symbol("LIT") == "LIGHTERUSDT"
    assert client.resolve_symbol("TRUMPSOL") == "TRUMPUSDT"


def test_binance_alias_never_falls_back_to_a_different_literal_instrument(monkeypatch):
    from event_engine.binance import BinanceMarketClient

    client = BinanceMarketClient()
    for logical_symbol, literal_symbol in (
        ("LIT", "LITUSDT"),
        ("TRUMPSOL", "TRUMPSOLUSDT"),
    ):
        monkeypatch.setattr(
            client, "refresh_exchange_info",
            lambda **kwargs: {literal_symbol: {"symbol": literal_symbol}},
        )
        assert client.resolve_symbol(logical_symbol) is None


def test_divergence_engine_flag_prevents_detector_execution(monkeypatch):
    import run_once as ro

    bucket = 100
    interval_ms = ro._timeframe_interval_ms("1h")
    rows = [
        {"open_time": i * interval_ms, "close_time": i * interval_ms + interval_ms - 1,
         "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}
        for i in range(bucket - 59, bucket + 1)
    ]
    monkeypatch.setattr(ro, "ENABLE_DIVERGENCE_ENGINE", False)
    monkeypatch.setattr(ro, "ENABLE_VOLUME_PROFILE_DIVERGENCE_ENGINE", False)
    monkeypatch.setattr(ro, "ENABLE_HARMONIC_PATTERN_ENGINE", False)
    for name in (
        "ENABLE_MACD_4H_ENGINE", "ENABLE_MA_COMPRESSION_ENGINE", "ENABLE_BREAKOUT_MOMENTUM_ENGINE",
        "ENABLE_DONCHIAN_RETEST_ENGINE", "ENABLE_LIQUIDITY_SWEEP_ENGINE", "ENABLE_EMA_PULLBACK_ENGINE",
        "ENABLE_ORDER_BLOCK_ENGINE", "ENABLE_BREAKER_BLOCK_ENGINE", "ENABLE_MITIGATION_BLOCK_ENGINE",
        "ENABLE_SFP_ENGINE", "ENABLE_LIQUIDATION_CASCADE_FVG_ENGINE", "ENABLE_CRT_ENGINE",
    ):
        monkeypatch.setattr(ro, name, False)
    monkeypatch.setattr(ro, "_symbol_scan_due", lambda *_args: True)
    monkeypatch.setattr(ro, "_fetch_market_klines_scan", lambda *_args: rows)
    monkeypatch.setattr(ro, "_scan_buckets_to_process", lambda *_args: [bucket])
    monkeypatch.setattr(ro, "_frame_through_bucket", lambda *_args: rows)
    monkeypatch.setattr(ro, "add_cvd", lambda frame: frame)
    monkeypatch.setattr(ro, "attach_oi_series", lambda frame, _history: frame)
    monkeypatch.setattr(ro, "attach_funding_series", lambda frame, _history: frame)
    monkeypatch.setattr(ro, "_load_oi_history", lambda: {})
    monkeypatch.setattr(ro, "_load_funding_history", lambda: {})
    monkeypatch.setattr(ro, "detect_divergences", lambda *_args: pytest.fail("disabled detector was called"))
    monkeypatch.setattr(ro, "ENABLE_VOLATILITY_SQUEEZE_ENGINE", False)
    monkeypatch.setattr(ro, "detect_squeeze_release", lambda *_args, **_kwargs: pytest.fail("disabled squeeze detector was called"))
    monkeypatch.setattr(ro, "_mark_symbol_scanned", lambda *_args: None)
    monkeypatch.setattr(ro, "_save_timeframe_scan_state", lambda *_args: None)
    monkeypatch.setattr(ro, "emit_event", lambda *_args: True)

    stats = {"events_total": 0, "divergence_events": 0, "squeeze_events": 0,
             "scan_errors": 0, "fresh_events": 0, "fresh_long": 0, "fresh_short": 0,
             "fresh_squeeze": 0, "fresh_divergence": 0}
    out = ro._refresh_timeframe_events(
        [SimpleNamespace(symbol="TEST")], "1h", 100, rows[-1]["close_time"],
        set(), stats, {"symbols": {}}, bucket,
    )
    assert out == []
    assert stats["divergence_events"] == 0


def test_divergence_data_dir_is_shared_by_runtime_modules(tmp_path):
    code = """
import json
import run_once
from event_engine import bingx, tracker, reconcile_daily
print(json.dumps({
  'run_once': str(run_once.DATA.resolve()),
  'tracker': str(tracker.DATA.resolve()),
  'bingx': str(bingx.OPEN_ORDERS_COOLDOWN_STATE_PATH.parent.resolve()),
  'reconcile': str(reconcile_daily.DATA_DIR.resolve()),
}))
"""
    env = dict(os.environ)
    env["DIVERGENCE_DATA_DIR"] = str(tmp_path / "missing-parent" / "nested" / "custom-runtime")
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env)
    paths = json.loads(result.stdout.strip().splitlines()[-1])
    expected_path = tmp_path / "missing-parent" / "nested" / "custom-runtime"
    expected = str(expected_path.resolve())
    assert set(paths.values()) == {expected}
    assert expected_path.is_dir()


def test_telegram_fact_formatting_handles_none_and_oversized_messages(monkeypatch):
    from event_engine import telegram

    for event in (
        {"direction": "LONG", "event_fact": {"p1_price": 100, "p2_price": 99, "price_delta_atr": None}},
        {"direction": "LONG", "event_fact": {"squeeze_duration_bars": 12, "compression_ratio": None}},
        {"direction": "SHORT", "event_fact": {"liq_ratio_24h": None, "spike_atr_mult": None}},
    ):
        assert isinstance(telegram.format_signal(event), str)

    captured = []
    monkeypatch.setenv("TG_BOT_TOKEN", "123456789:DUMMY_TOKEN")
    monkeypatch.setenv("TG_CHAT_IDS", "A")

    class Response:
        status_code = 200

        def json(self):
            return {"ok": True, "result": {"message_id": 1}}

    monkeypatch.setattr(telegram.requests, "post", lambda url, **kwargs: captured.append(kwargs) or Response())
    outcome = telegram.send_detailed("<b>" + ("x" * 5000) + "</b>")
    assert outcome["A"]["sent"] is True
    assert len(captured[0]["data"]["text"]) <= 4096
    assert "parse_mode" not in captured[0]["data"]


def test_telegram_429_retry_after_is_returned_for_durable_per_chat_retry(monkeypatch, tmp_path):
    from event_engine import telegram
    from event_engine import tracker

    monkeypatch.setenv("TG_BOT_TOKEN", "123456789:DUMMY_TOKEN")
    monkeypatch.setenv("TG_CHAT_IDS", "A")
    calls = []

    class Response:
        status_code = 429

        def json(self):
            return {"ok": False, "error_code": 429, "description": "Too Many Requests",
                    "parameters": {"retry_after": 30}}

    monkeypatch.setattr(telegram.requests, "post", lambda *_args, **_kwargs: calls.append(1) or Response())
    result = telegram.send_detailed("message")
    assert result["A"]["sent"] is False
    assert result["A"]["retry_after_sec"] == 30
    assert len(calls) == 1

    path = tmp_path / "notifications.json"
    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", path)
    now = {"value": 1_000.0}
    monkeypatch.setattr(tracker.time, "time", lambda: now["value"])
    deliveries = iter([
        {"A": {"sent": False, "error": "429", "retry_after_sec": 30}},
        {"A": {"sent": True, "message_id": 77}},
    ])
    monkeypatch.setattr(tracker, "send_detailed", lambda *_args, **_kwargs: next(deliveries))
    assert tracker._queue_notification(
        "TG_RETRY_AFTER", event_id="EVT_RETRY", kind="TRADE_CLOSE",
        symbol="BTC-USDT", direction="LONG", text="close",
    ) is False
    state = json.loads(path.read_text(encoding="utf-8"))["TG_RETRY_AFTER"]["chats"]["A"]
    assert state["retry_after_ts"] == 1_030_000

    now["value"] = 1_020.0
    tracker._retry_pending_notifications()
    assert json.loads(path.read_text(encoding="utf-8"))["TG_RETRY_AFTER"]["chats"]["A"]["sent"] is False

    now["value"] = 1_031.0
    tracker._retry_pending_notifications()
    final_state = json.loads(path.read_text(encoding="utf-8"))["TG_RETRY_AFTER"]
    assert final_state["resolved"] is True
    assert final_state["chats"]["A"]["message_id"] == 77


def test_nested_corrupt_notification_chat_state_is_not_overwritten(tmp_path, monkeypatch):
    from event_engine import tracker

    path = tmp_path / "notifications.json"
    original = b'{"TG_BAD":{"text":"x","chats":{"A":"not-an-object"}}}'
    path.write_bytes(original)
    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", path)

    with pytest.raises(tracker.NotificationStateError, match="non-object chat states"):
        tracker._load_notifications()
    assert path.read_bytes() == original


def test_active_trade_protection_update_avoids_identical_write_churn(tmp_path, monkeypatch):
    from event_engine import tracker

    path = tmp_path / "active_trades.json"
    initial = {
        "EVT": {
            "event_id": "EVT", "symbol": "TEST-USDT", "direction": "LONG", "closed": False,
            "tp_fill_ledger_schema_version": 1, "tp_filled_qty_by_order": {"TP1": 0.0},
            "tp_filled_qty": {}, "tp_orders": [{"order_id": "TP1", "leg": "tp1"}],
            "sl_order": {"order_id": "SL1", "stop_price": 93.0},
        }
    }
    original = json.dumps(initial, ensure_ascii=False, indent=2)
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", path)
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda _symbol: "TEST-USDT")
    monkeypatch.setattr(tracker, "_save_active_trades", lambda *_args: pytest.fail("identical protection caused a write"))

    assert tracker.update_active_trade_protection(
        "TEST-USDT", "LONG", [{"order_id": "TP1", "leg": "tp1"}],
        {"order_id": "SL1", "stop_price": 93.0},
    ) is True
    assert path.read_text(encoding="utf-8") == original


def test_coinalyze_pagination_continues_after_visible_numbered_page_window(monkeypatch):
    import playwright.sync_api as sync_api
    from event_engine import coinalyze

    class Page:
        current = 1

        def query_selector(self, selector):
            return object() if selector == ".pagination" else None

        def content(self):
            return f"page-{self.current}"

        def wait_for_selector(self, *_args, **_kwargs):
            pass

        def wait_for_timeout(self, *_args, **_kwargs):
            pass

    class Browser:
        def close(self):
            pass

    class Context:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return False

    page = Page()
    monkeypatch.setattr(sync_api, "sync_playwright", lambda: Context())
    monkeypatch.setattr(coinalyze, "_setup_browser_context", lambda _p: (Browser(), page))
    monkeypatch.setattr(coinalyze, "_load_page", lambda _page, url: "page-1" if url == coinalyze.COINALYZE_URL else "page-2")
    monkeypatch.setattr(coinalyze, "get_page_urls", lambda _html: [coinalyze.COINALYZE_URL, "page2-url"])
    monkeypatch.setattr(coinalyze, "parse_table", lambda html: [SimpleNamespace(symbol={"page-1": "AAA", "page-2": "BBB", "page-3": "CCC"}[html])])
    clicks = []

    def click_next(_page, current_page):
        clicks.append(current_page)
        if current_page == 2:
            page.current = 3
            return True
        return False

    monkeypatch.setattr(coinalyze, "click_next_page", click_next)
    result = coinalyze.fetch_data()

    assert [row.symbol for row in result] == ["AAA", "BBB", "CCC"]
    assert clicks == [2, 3]


def test_coinalyze_page_url_extraction_excludes_next_previous_and_ellipsis():
    from event_engine.coinalyze import _get_numbered_page_urls, COINALYZE_URL

    html = '''
    <div class="pagination">
      <a href="/prev" aria-label="Previous">«</a>
      <a href="/page2">2</a>
      <a href="/page3">3</a>
      <a href="/page4" title="Page 4">4</a>
      <span>…</span>
      <a href="/next" aria-label="Next">Next</a>
    </div>
    '''
    urls = _get_numbered_page_urls(html)
    assert urls == {
        2: "https://coinalyze.net/page2",
        3: "https://coinalyze.net/page3",
        4: "https://coinalyze.net/page4",
    }


@pytest.mark.parametrize(
    "chat_state",
    [
        {"sent": "false", "attempts": 0},
        {"sent": False, "attempts": -1},
        {"sent": False, "attempts": 0, "retry_after_ts": "NaN"},
    ],
)
def test_corrupt_notification_delivery_fields_are_rejected(chat_state, tmp_path, monkeypatch):
    from event_engine import tracker

    path = tmp_path / "notifications.json"
    path.write_text(json.dumps({"TG_BAD": {"text": "x", "chats": {"A": chat_state}}}), encoding="utf-8")
    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", path)
    original = path.read_bytes()
    with pytest.raises(tracker.NotificationStateError):
        tracker._load_notifications()
    assert path.read_bytes() == original


def test_compound_cvd_liquidation_veto_is_long_only(monkeypatch):
    from types import SimpleNamespace
    import run_once as ro

    monkeypatch.setattr(ro, "ENTRY_QUALITY_GATE_ENABLED", True)
    monkeypatch.setattr(ro, "ENTRY_WEAK_ENGINE_BLOCK_ENABLED", False)
    monkeypatch.setattr(ro, "SHORT_OI_VETO_ENABLED", False)
    monkeypatch.setattr(ro, "COMPOUND_CVD_LIQ_VETO_ENABLED", True)
    monkeypatch.setattr(ro, "COMPOUND_CVD_MAX", 25.0)
    monkeypatch.setattr(ro, "COMPOUND_LIQ_LONG_MIN", 150_000.0)
    monkeypatch.setattr(ro, "SYMBOL_LOSS_COOLDOWN_MIN", 0.0)
    row = SimpleNamespace(oi_chg24_pct=10.0, cvd24=10.0, liq_long24=200_000.0)

    long_result = ro._entry_quality_gate(
        ev={"symbol": "TEST", "event_type": "DONCHIAN_RETEST_BREAKOUT"},
        row=row, direction="LONG", now_ms=1_000_000, recent_loss_ts={},
    )
    short_result = ro._entry_quality_gate(
        ev={"symbol": "TEST", "event_type": "DONCHIAN_RETEST_BREAKOUT"},
        row=row, direction="SHORT", now_ms=1_000_000, recent_loss_ts={},
    )
    assert long_result[0] is False
    assert any(reason.startswith("CVD_LIQ:") for reason in long_result[1])
    assert short_result == (True, [], False)
