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


def test_release_uses_binance_for_signal_market_data_and_bingx_for_execution():
    workflow = _workflow()
    assert 'MARKET_DATA_SOURCE: binance' in workflow
    assert 'BINGX_BASE_URL: https://open-api-vst.bingx.com' in workflow
    assert 'EXECUTION_MODE: vst' in workflow


def test_release_enables_requested_signal_engines():
    workflow = _workflow()
    assert 'DIVERGENCE_VOLUME_CONFIRMATION_ENABLED: "true"' in workflow
    assert 'ENABLE_VOLUME_PROFILE_DIVERGENCE_ENGINE: "true"' in workflow
    assert 'ENABLE_HARMONIC_PATTERN_ENGINE: "true"' in workflow


def test_release_has_cross_exchange_price_guard():
    workflow = _workflow()
    assert 'CROSS_EXCHANGE_PRICE_GUARD_ENABLED: "true"' in workflow
    assert 'MAX_CROSS_EXCHANGE_DRIFT_PCT: "1.00"' in workflow


def test_release_pins_runner_and_installs_userspace_wireproxy():
    workflow = _workflow()
    assert "runs-on: ubuntu-24.04" in workflow
    assert "Install pinned Binance WireProxy" in workflow
    assert "v1.1.3" in workflow
    assert "wireproxy_linux_amd64.tar.gz" in workflow
    assert "e88c1d090740373fc606c1bafd81d9a5eadc642cce5667616e20e9d7a444f51c" in workflow
    assert "--configtest" in workflow
    assert "wg-quick up wg0" not in workflow
    assert "wg-quick down wg0" not in workflow
    assert "sudo wg" not in workflow


def test_release_has_binance_proxy_preflight_before_engine_and_cleanup_before_commit():
    workflow = _workflow()
    assert "Start Binance-only WireProxy" in workflow
    assert "Binance Futures preflight" in workflow
    assert '"$BASE/fapi/v1/exchangeInfo"' in workflow
    assert '"$BASE/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=10"' in workflow
    assert '"$BASE/fapi/v1/ticker/price?symbol=BTCUSDT"' in workflow
    assert "Stop Binance WireProxy" in workflow
    assert 'BINANCE_VPN_ENABLED: "true"' in workflow
    assert 'BINANCE_HTTP_PROXY: http://127.0.0.1:18080' in workflow
    assert workflow.index("Start Binance-only WireProxy") < workflow.index("Binance Futures preflight")
    assert workflow.index("Binance Futures preflight") < workflow.index("Run engine")
    assert workflow.index("Stop Binance WireProxy") < workflow.index("Commit state")


def test_release_requires_wireguard_secret_and_country_guard():
    workflow = _workflow()
    assert 'WIREGUARD_CONF: ${{ secrets.WIREGUARD_CONF }}' in workflow
    assert 'VPN_EXPECTED_COUNTRY: NL' in workflow
    assert 'actual != expected.upper()' in workflow


def test_release_has_manual_vpn_test_workflow():
    from pathlib import Path

    workflow = Path('.github/workflows/vpn-test.yml')
    assert workflow.exists()
    text = workflow.read_text(encoding='utf-8')
    assert 'workflow_dispatch:' in text
    assert 'wg-quick up wg0' not in text
    assert 'wg-quick down wg0' not in text
    assert 'Install pinned Binance WireProxy' in text
    assert 'wireproxy_linux_amd64.tar.gz' in text
    assert 'e88c1d090740373fc606c1bafd81d9a5eadc642cce5667616e20e9d7a444f51c' in text
    assert 'BASE="${BINANCE_BASE_URL%/}"' in text
    assert '"$BASE/fapi/v1/exchangeInfo"' in text
    assert '"$BASE/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=10"' in text
    assert '"$BASE/fapi/v1/klines?symbol=BTCUSDT&interval=4h&limit=10"' in text
    assert 'expected_request_count = 22' in text
    assert "Manifest request count mismatch" in text
    assert 'Duplicate request labels detected in manifest' in text
    assert 'test "$rows" -eq 22' in text
    assert 'WARNING: expected 22 request rows' not in text
    assert 'bingx_live_price_all' in text
    assert '$BINGX_BASE_URL/openApi/swap/v2/quote/price' in text
    assert 'bingx_live_price_all.body' in text
    assert 'Cross-exchange live price drift >1%' in text
    assert 'latest returned candle' not in text
    assert '"$BASE/fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=10"' in text
