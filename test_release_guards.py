from pathlib import Path

WORKFLOW = Path('.github/workflows/event-engine.yml')


def _workflow() -> str:
    return WORKFLOW.read_text(encoding='utf-8')


def test_clean_release_uses_divergence_shadow_mode():
    assert 'DIVERGENCE_SHADOW_ONLY: "true"' in _workflow()


def test_clean_release_has_pre_order_drift_retry_budget():
    assert 'MAX_PRE_ORDER_DRIFT_REJECTIONS: "3"' in _workflow()


def test_shadow_flag_is_actually_reachable():
    """A config guard is worthless unless the flag can fire.

    DIVERGENCE_SHADOW_ONLY shipped as "true" for a full week while the branch
    guarding it compared event_type to the literal "DIVERGENCE", which no
    detector emits. Assert the predicate, not the YAML string.
    """
    import run_once

    assert 'DIVERGENCE_SHADOW_ONLY: "true"' in _workflow()
    assert run_once._is_divergence_event(
        {"event_type": "REGULAR_BULLISH_RSI", "event_fact": {"engine": "DIVERGENCE"}}
    )


def test_mitigation_engine_is_disabled():
    """Negative in both production windows under both directional settings."""
    assert 'ENABLE_MITIGATION_BLOCK_ENGINE: "false"' in _workflow()


def test_break_even_policy_matches_documented_ladder():
    import event_engine.tracker as tracker

    assert 'BE_AFTER_LEG: "tp2"' in _workflow()
    assert tracker.BE_AFTER_LEG == "tp2"


def test_liquidity_sweep_has_a_retest_window():
    assert 'LIQUIDITY_SWEEP_RETEST_MAX_DELAY_MIN' in _workflow()


def test_kline_history_warms_ema200():
    """EMA200 is a hard regime gate; 250 bars leaves ~8% seed weight."""
    workflow = _workflow()
    for key in ("KLINE_LIMIT_1H", "KLINE_LIMIT_4H", "KLINE_LIMIT_1D"):
        line = next(l for l in workflow.splitlines() if l.strip().startswith(key + ":"))
        assert int(line.split('"')[1]) >= 400, line


def test_kline_rate_limit_circuit_breaker_settings():
    workflow = _workflow()
    assert 'BINGX_KLINE_SCAN_MIN_INTERVAL_SEC: "1.25"' in workflow
    assert 'BINGX_KLINE_RETRY_ATTEMPTS: "3"' in workflow
    assert 'BINGX_KLINE_RATE_LIMIT_FALLBACK_SEC: "900"' in workflow
