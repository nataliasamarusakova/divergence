import pytest

from event_engine import tracker


def test_corrupt_notification_outbox_is_not_treated_as_empty(tmp_path, monkeypatch):
    path = tmp_path / "notifications.json"
    original = b'{"pending":BROKEN\n'
    path.write_bytes(original)
    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", path)

    with pytest.raises(tracker.NotificationStateError):
        tracker._load_notifications()

    assert path.read_bytes() == original


def test_notification_retry_does_not_overwrite_corrupt_outbox(tmp_path, monkeypatch):
    path = tmp_path / "notifications.json"
    original = b'{"pending":BROKEN\n'
    path.write_bytes(original)
    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", path)

    tracker._retry_pending_notifications()

    assert path.read_bytes() == original


def test_notification_enqueue_refuses_to_overwrite_corrupt_outbox(tmp_path, monkeypatch):
    path = tmp_path / "notifications.json"
    original = b'{"pending":BROKEN\n'
    path.write_bytes(original)
    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", path)

    with pytest.raises(tracker.NotificationStateError):
        tracker._queue_notification(
            "TG_TEST", event_id="EVT_TEST", kind="TRADE_CLOSE",
            symbol="BTC-USDT", direction="LONG", text="test",
        )

    assert path.read_bytes() == original


def test_non_object_notification_record_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "notifications.json"
    original = b'{"TG_BAD": []}'
    path.write_bytes(original)
    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", path)

    with pytest.raises(tracker.NotificationStateError):
        tracker._load_notifications()

    assert path.read_bytes() == original


def test_protection_update_does_not_mutate_quarantine_sentinels(tmp_path, monkeypatch):
    import json

    path = tmp_path / "active_trades.json"
    original = {
        "ROLLBACK_CLEANUP_TEST": {
            "event_id": "ROLLBACK_CLEANUP_TEST", "symbol": "BTC-USDT",
            "direction": "LONG", "closed": False,
            "protection_cleanup_only": True, "close_cleanup_pending": True,
            "remaining_qty": 0.0,
        },
        "ENTRY_PENDING_TEST": {
            "event_id": "ENTRY_PENDING_TEST", "symbol": "ETH-USDT",
            "direction": "SHORT", "closed": False,
            "entry_order_resolution_pending": True, "close_cleanup_pending": True,
            "remaining_qty": 0.0,
        },
    }
    path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", path)

    assert tracker.update_active_trade_protection(
        "BTC-USDT", "LONG", [{"leg": "tp1", "order_id": "TP1"}],
        {"order_id": "SL1"},
    ) is False
    assert tracker.update_active_trade_protection(
        "ETH-USDT", "SHORT", [{"leg": "tp1", "order_id": "TP2"}],
        {"order_id": "SL2"},
    ) is False
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_reconciliation_does_not_match_cleanup_only_record_as_live_trade():
    import run_once as ro

    active = {
        "ROLLBACK_CLEANUP_TEST": {
            "event_id": "ROLLBACK_CLEANUP_TEST", "symbol": "BTC-USDT",
            "direction": "LONG", "closed": False,
            "protection_cleanup_only": True, "close_cleanup_pending": True,
        }
    }
    assert ro._find_active_trade_for_position("BTC-USDT", "LONG", active) is None
