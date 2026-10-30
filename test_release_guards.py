from pathlib import Path

WORKFLOW = Path('.github/workflows/event-engine.yml')


def _workflow() -> str:
    return WORKFLOW.read_text(encoding='utf-8')


def test_clean_release_executes_divergence_events():
    assert 'DIVERGENCE_SHADOW_ONLY: "false"' in _workflow()


def test_v10418_event_policy_keeps_requested_families_enabled_and_shadow_only():
    workflow = _workflow()
    assert 'DISABLED_EVENT_TYPES: ""' in workflow
    assert 'ENTRY_QUALITY_MODE: "shadow"' in workflow
    assert 'CRT_BULLISH,CRT_BEARISH,MA_COMPRESSION_BREAKOUT,EMA_PULLBACK_CONTINUATION,LIQUIDITY_SWEEP_RECLAIM,VOLUME_PROFILE_DISTRIBUTION,BREAKER_BLOCK_BEARISH' in workflow
    assert 'ENABLE_BREAKOUT_MOMENTUM_ENGINE: "true"' in workflow
    import run_once
    assert not run_once._event_type_is_disabled("CRT_BEARISH")
    assert not run_once._event_type_is_disabled("CRT_BULLISH")
    assert not run_once._event_type_is_disabled("BREAKOUT_MOMENTUM")
    assert not run_once._event_type_is_disabled("BREAKER_BLOCK_BULLISH")
    assert 'SHORT_OI_VETO_ENABLED:' not in workflow
    assert 'SHORT_OI_VETO_PCT:' not in workflow


def test_clean_release_has_pre_order_drift_retry_budget():
    assert 'MAX_PRE_ORDER_DRIFT_REJECTIONS: "3"' in _workflow()
    assert 'MAX_CROSS_EXCHANGE_DRIFT_REJECTIONS: "3"' in _workflow()
    assert 'ENTRY_QUALITY_GATE_ENABLED: "true"' in _workflow()
    assert 'ENTRY_QUALITY_MODE: "shadow"' in _workflow()
    assert 'REQUIRE_MULTI_TP: "false"' in _workflow()
    assert 'REJECT_SINGLE_TP: "false"' in _workflow()
    assert 'MIN_SETUP_SCORE: "60"' in _workflow()
    assert 'MIN_SHORT_SETUP_SCORE: "75"' in _workflow()
    assert 'REJECT_ATR_RISK_CLIP: "false"' in _workflow()
    assert 'FIXED_STOP_LOSS_PCT: "7.00"' in _workflow()


def test_vst_research_release_keeps_portfolio_and_cycle_caps_off():
    workflow = _workflow()
    assert 'EXECUTION_MODE: vst' in workflow
    assert 'PORTFOLIO_CAP_ENABLED: "false"' in workflow
    assert 'MAX_TRADES_PER_CYCLE: "0"' in workflow


def test_scheduler_state_uses_v9_style_persistent_data_without_pre_run_cleanup():
    workflow = _workflow()
    assert 'data/timeframe_scan_state.json' in workflow
    assert 'recent_event_cache.json' not in workflow  # runtime path is owned by run_once.py
    assert 'git add -f -- data/timeframe_scan_state.json' in workflow
    assert 'git add -A -- data/' in workflow
    assert 'git diff --quiet' in workflow
    assert 'git fetch --prune --no-tags origin main' in workflow
    assert 'merged scheduler state' in workflow
    assert 'Synchronize persistent scheduler state before tests' not in workflow
    assert 'Backup persisted runtime state before tests' not in workflow
    assert 'Clean test-generated runtime state' not in workflow


def test_scheduler_state_push_is_verified_on_remote():
    workflow = _workflow()
    assert '[COMMIT_STATE] scheduler state version=2 symbols=' in workflow
    assert 'remote_sha256=' in workflow
    assert 'remote scheduler state verified symbols=' in workflow
    assert 'State pushed and verified successfully' in workflow
    assert 'REBASE_HEAD' in workflow
    assert 'Resolving scheduler-state conflict with max-watermark merge' in workflow
    assert 'GIT_EDITOR=true git rebase --continue' in workflow


def test_release_telemetry_uses_attempts_and_opened_trades_separately():
    import run_once
    source = Path(run_once.__file__).read_text(encoding="utf-8")
    assert "ENGINE_SUMMARY] trades_this_cycle=" not in source
    assert "CYCLE END: trades_this_cycle=" not in source
    assert "opened_trades_this_cycle=" in source
    assert "execution_attempts_this_cycle=" in source


def test_divergence_predicate_is_reachable_for_real_execution():
    """A config guard is worthless unless the flag can fire.

    Divergence events are recognised through their engine/event semantics rather
    than a literal event_type equal to "DIVERGENCE".
    """
    import run_once

    assert 'DIVERGENCE_SHADOW_ONLY: "false"' in _workflow()
    assert run_once._is_divergence_event(
        {"event_type": "REGULAR_BULLISH_RSI", "event_fact": {"engine": "DIVERGENCE"}}
    )


def test_mitigation_engine_is_disabled():
    """Negative in both production windows under both directional settings."""
    assert 'ENABLE_MITIGATION_BLOCK_ENGINE: "false"' in _workflow()


def test_break_even_policy_matches_documented_ladder():
    import event_engine.tracker as tracker

    assert 'BE_AFTER_LEG: "tp1"' in _workflow()
    assert tracker.BE_AFTER_LEG == "tp1"


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


def test_release_wires_trend_decision_to_closed_trigger_timestamp():
    source = Path("run_once.py").read_text(encoding="utf-8")
    assert "decision_ts = _trend_decision_timestamp(trigger_diag, detected_at, now_ms)" in source
    assert "def _trend_decision_timestamp(" in source
    assert 'decision_ts = int(trigger_observed_at_ts or time.time() * 1000)' not in source


def test_trend_decision_timestamp_falls_back_when_15m_trigger_is_not_required():
    import run_once

    detected_at = 1_760_000_000_000
    cycle_now = detected_at + 60_000
    assert run_once._trend_decision_timestamp({"ok": True, "reason": "not_required"}, detected_at, cycle_now) == detected_at
    assert run_once._trend_decision_timestamp({"trigger_bar_close_ts": detected_at - 900_000}, detected_at, cycle_now) == detected_at - 900_000
    assert run_once._trend_decision_timestamp({"reason": "not_required"}, 0, cycle_now) == cycle_now


def test_release_uses_fixed_seven_percent_stop():
    workflow = _workflow()
    assert 'FIXED_STOP_LOSS_PCT: "7.00"' in workflow
    assert 'TREND_FILTER_ENABLED: "true"' in workflow
    assert 'TREND_FILTER_MODE: "shadow"' in workflow
    assert 'TREND_FILTER_MIN_1H_BARS: "400"' in workflow
    assert 'TREND_FILTER_MIN_4H_BARS: "400"' in workflow
    assert 'TREND_FILTER_REQUIRE_PERSISTENCE: "false"' in workflow
    assert 'SR_3H_DIRECTIONAL_GATE_ENABLED = True' in Path("run_once.py").read_text(encoding="utf-8")
    assert 'SR_DIRECTIONAL_MAX_DISTANCE_PCT = 5.0' in Path("event_engine/sr_context.py").read_text(encoding="utf-8")
    assert 'AJAY_SR_ROOM_ENABLED: "false"' in workflow
    assert 'AJAY_SR_ROOM_MODE: "off"' in workflow
    assert 'MAX_ENTRY_RISK_PCT: "7.00"' in workflow
    assert 'REJECT_ATR_RISK_CLIP: "false"' in workflow


def test_release_has_no_legacy_sr_tp3_cap_setting():
    workflow = _workflow()
    assert "AJAY_SR_MIN_PARTIAL_TP3_R" not in workflow
    assert "apply_sr_tp3_cap" not in workflow


def test_release_enforces_directional_3h_spot_gate_and_keeps_optional_room_off():
    workflow = _workflow()
    assert 'SR_3H_DIRECTIONAL_GATE_ENABLED = True' in Path("run_once.py").read_text(encoding="utf-8")
    assert 'SR_DIRECTIONAL_MAX_DISTANCE_PCT = 5.0' in Path("event_engine/sr_context.py").read_text(encoding="utf-8")
    assert 'AJAY_SR_ROOM_ENABLED: "false"' in workflow
    assert 'AJAY_SR_ROOM_MODE: "off"' in workflow
    assert 'AJAY_SR_REQUIRE_DATA: "true"' in workflow
    assert 'AJAY_SR_SPOT_BASE_URL: "https://data-api.binance.vision"' in workflow
    assert 'AJAY_SR_HTTP_PROXY: "http://127.0.0.1:18080"' in workflow
    assert 'AJAY_SR_KLINE_LIMIT_1H: "1000"' in workflow
    assert 'AJAY_SR_MAX_DATA_AGE_MIN: "190"' in workflow
    assert 'AJAY_SR_REQUEST_MIN_INTERVAL_SEC: "0.10"' in workflow
    assert 'AJAY_SR_RB: "10"' in workflow
    assert 'AJAY_SR_PRD: "284"' in workflow


def test_pine_sr_manual_diagnostic_workflow_is_spot_only():
    wf = Path('.github/workflows/pine-sr-diagnostic.yml').read_text(encoding='utf-8')
    assert 'workflow_dispatch' in wf
    assert 'pine_sr_diagnostic.py' in wf
    assert 'BTC-USDT' in wf


def test_release_has_mandatory_3h_direction_gate_and_cached_spot_snapshot():
    workflow = _workflow()
    runtime = Path('run_once.py').read_text(encoding='utf-8')
    sr_source = Path('event_engine/sr_context.py').read_text(encoding='utf-8')
    assert 'SR_3H_DIRECTIONAL_GATE_ENABLED = True' in runtime
    assert 'get_cached_sr_snapshot' in runtime
    assert 'evaluate_sr_directional_zone' in runtime
    assert 'interval="1h"' in sr_source
    assert '_aggregate_closed_3h_candles' in sr_source
    assert 'AJAY_SR_SPOT_BASE_URL: "https://data-api.binance.vision"' in workflow


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


def test_release_keeps_only_required_binance_proxy_stage_and_no_redundant_futures_preflight():
    workflow = _workflow()
    assert "Start Binance-only WireProxy" in workflow
    assert "Binance Futures preflight" not in workflow
    assert "Stop Binance WireProxy" in workflow
    assert 'BINANCE_VPN_ENABLED: "true"' in workflow
    assert 'BINANCE_HTTP_PROXY: http://127.0.0.1:18080' in workflow
    assert workflow.index("Start Binance-only WireProxy") < workflow.index("Run engine")
    assert workflow.index("Run engine") < workflow.index("Stop Binance WireProxy")
    assert workflow.index("Stop Binance WireProxy") < workflow.index("Commit state")


def test_release_removes_per_cycle_test_and_compile_stages():
    workflow = _workflow()
    assert "- name: Run tests" not in workflow
    assert "- name: Syntax check" not in workflow


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
    assert 'mkdir -p vpn-audit/responses vpn-audit/pretty' in text
    assert "label|url|http_status|curl_exit|body_sha|body_bytes\\n' > vpn-audit/manifest.tsv" in text
    assert text.index('mkdir -p vpn-audit/responses vpn-audit/pretty') < text.index('tee vpn-audit/dns-routing.txt')


def test_pytest_isolates_production_sr_environment(monkeypatch):
    """CI's workflow env must not force S/R onto unrelated unit tests."""
    import conftest
    assert hasattr(conftest, "isolate_production_entry_gates")


def test_release_has_live_spot_symbol_resolver_and_persisted_sr_budget():
    workflow = _workflow()
    assert 'AJAY_SR_SPOT_EXCHANGE_INFO_TTL_SEC: "900"' in workflow
    assert 'MAX_SR_DATA_REJECTIONS: "3"' in workflow
    source = Path('event_engine/sr_context.py').read_text(encoding='utf-8')
    assert 'def resolve_spot_symbol' in source
    assert 'SRSymbolUnavailableError' in source
    assert 'source_price_scale' in source
    assert 'source_volume_scale' in source


def test_release_does_not_use_direct_spot_klines_normalization_in_diagnostic():
    source = Path('pine_sr_diagnostic.py').read_text(encoding='utf-8')
    assert 'resolve_spot_symbol' in source
    assert 'get_cached_sr_snapshot' in source
    assert 'KLINES_URL' not in source
    assert 'normalize_symbol' not in source


def test_new_entries_fail_closed_when_bingx_catalog_refresh_fails():
    source = Path('run_once.py').read_text(encoding='utf-8')
    assert 'bingx_contract_catalog_fresh = False' in source
    assert 'NEW ENTRIES BLOCKED for this cycle' in source
    assert 'if not bingx_contract_catalog_fresh:' in source
    assert 'stats["rejected_bingx_contract"]' in source


def test_release_never_deletes_persistent_data_between_engine_runs():
    workflow = _workflow()
    assert 'rm -rf data' not in workflow
    assert 'Backup persisted runtime state before tests' not in workflow
    assert 'Clean test-generated runtime state (preserve persisted data)' not in workflow
    assert 'Run tests' not in workflow
    assert workflow.index('Checkout') < workflow.index('Run engine') < workflow.index('Commit state')


def test_release_rejects_stale_trigger_before_scoring_and_opportunity_queue():
    source = Path("run_once.py").read_text(encoding="utf-8")
    trigger_success = source.index('if not trigger_diag.get("ok"):')
    freshness_check = source.index('_trigger_freshness_status(', trigger_success)
    score_calculation = source.index('score = calculate_setup_score', freshness_check)
    setup_pass = source.index('stage="STRATEGY_FILTERS", decision="PASSED"', score_calculation)
    assert trigger_success < freshness_check < score_calculation < setup_pass
    assert 'reason = "TRIGGER_STALE_AT_CANDIDATE_EVALUATION"' in source
    assert 'TRIGGER_TIMESTAMP_UNAVAILABLE' in source[freshness_check:score_calculation]


def test_candidate_decision_journal_uses_actual_stage_timestamp():
    source = Path("run_once.py").read_text(encoding="utf-8")
    candidate_loop = source.index('stage="SIGNAL_DISCOVERED", decision="SEEN"')
    assert 'evaluation_ts=int(time.time() * 1000)' in source[candidate_loop:candidate_loop + 300]
    reject_helper = source.index('def reject_candidate(stage: str, reason: str, **details: Any) -> None:')
    reject_end = source.index('if not event_id:', reject_helper)
    assert 'evaluation_ts=int(time.time() * 1000)' in source[reject_helper:reject_end]


def test_research_unlimited_entries_are_unlimited_without_hidden_entry_throttles():
    workflow = _workflow()
    # The mode is inferred from EXECUTION_MODE=vst, not explicitly pinned with a
    # second flag that could drift out of sync. Conflicting numeric caps are not set.
    assert 'EXECUTION_MODE: vst' in workflow
    assert 'RESEARCH_UNLIMITED_ENTRIES:' not in workflow
    assert 'MAX_TRADES_PER_CYCLE: "0"' in workflow
    assert 'PORTFOLIO_CAP_ENABLED: "false"' in workflow
    assert 'SYMBOL_ENTRY_COOLDOWN_MIN: "0"' in workflow
    assert 'SQUEEZE_SYMBOL_ENTRY_COOLDOWN_MIN: "0"' in workflow
    assert 'SYMBOL_LOSS_COOLDOWN_MIN: "0"' in workflow
    assert 'SYMBOL_MAX_CONSECUTIVE_LOSSES: "0"' in workflow
    assert 'SYMBOL_QUARANTINE_MIN: "0"' in workflow
    for cap in ('MAX_ACTIVE_TRADES:', 'MAX_ACTIVE_LONGS:', 'MAX_ACTIVE_SHORTS:'):
        assert cap not in workflow
    blocked_line = next(line for line in workflow.splitlines() if 'ENTRY_BLOCKED_EVENT_TYPES:' in line)
    assert 'MA_COMPRESSION_BREAKOUT' in blocked_line
    runtime_source = Path('run_once.py').read_text(encoding='utf-8')
    assert 'MA_COMPRESSION_BREAKOUT,EMA_PULLBACK_CONTINUATION,LIQUIDITY_SWEEP_RECLAIM,VOLUME_PROFILE_DISTRIBUTION' in runtime_source
    assert 'MAX_ACTIVE_TRADES = 0' in runtime_source
    assert 'MAX_ACTIVE_LONGS = 0' in runtime_source
    assert 'MAX_ACTIVE_SHORTS = 0' in runtime_source
    import run_once
    assert run_once.RESEARCH_UNLIMITED_ENTRIES is True
    assert run_once.MAX_TRADES == 0
    assert run_once.PORTFOLIO_CAP_ENABLED is False
    assert run_once.MAX_ACTIVE_TRADES == run_once.MAX_ACTIVE_LONGS == run_once.MAX_ACTIVE_SHORTS == 0
    assert run_once.SYMBOL_LOSS_COOLDOWN_MIN == 0

    assert 'EARLY_LOSS_CUT_ENABLED: "false"' in workflow
    assert 'EARLY_LOSS_CUT_2H_MIN: "120"' in workflow
    assert 'EARLY_LOSS_CUT_2H_PNL: "-1.50"' in workflow
    assert 'EARLY_LOSS_CUT_4H_MIN: "240"' in workflow
    assert 'EARLY_LOSS_CUT_4H_PNL: "-2.50"' in workflow
    assert 'LONG_LS_GATE_ENABLED: "false"' in workflow
    assert 'LONG_LS_MAX: "1.36"' in workflow
    assert 'LONG_OI_CHG4H_MAX: "3.0"' in workflow
    assert 'LONG_FR_BAND_MIN: "0.0009"' in workflow
    assert 'LONG_FR_BAND_MAX: "0.00665"' in workflow
    assert 'LONG_BTC_VETO_ENABLED: "false"' in workflow
    assert 'TREND_FILTER_MODE: "shadow"' in workflow


def test_long_gate_missing_metrics_do_not_veto_and_explicit_thresholds_do():
    from types import SimpleNamespace
    import run_once

    run_once.LONG_LS_GATE_ENABLED = True
    run_once.LONG_LS_MAX = 1.36
    run_once.LONG_OI_CHG4H_MAX = 3.0
    run_once.LONG_FR_BAND_MIN = 0.0009
    run_once.LONG_FR_BAND_MAX = 0.00665
    assert run_once._long_market_risk_gate_reasons(SimpleNamespace(ls_accounts=None, oi_chg4h_pct=None, fr_oiw=None)) == []
    assert run_once._long_market_risk_gate_reasons(SimpleNamespace(ls_accounts=1.36, oi_chg4h_pct=3.0, fr_oiw=0.0008)) == []
    reasons = run_once._long_market_risk_gate_reasons(SimpleNamespace(ls_accounts=1.37, oi_chg4h_pct=3.1, fr_oiw=0.001))
    assert any(reason.startswith("LONG_LS:") for reason in reasons)
    assert any(reason.startswith("LONG_OI4H:") for reason in reasons)
    assert any(reason.startswith("LONG_FR:") for reason in reasons)


def test_long_btc_regime_gate_fails_closed_only_for_long_and_handles_btc_self():
    import run_once

    assert run_once._long_btc_regime_veto_reason("LONG", {"btc_regime": "BULLISH"}, enabled=True) is None
    assert run_once._long_btc_regime_veto_reason("LONG", {"btc_regime": "BEARISH"}, enabled=True)
    assert run_once._long_btc_regime_veto_reason("LONG", {"btc_regime": "MIXED"}, enabled=True)
    assert run_once._long_btc_regime_veto_reason("LONG", {"btc_regime": "UNKNOWN"}, enabled=True)
    assert run_once._long_btc_regime_veto_reason("LONG", None, enabled=True)
    assert run_once._long_btc_regime_veto_reason("SHORT", {"btc_regime": "BEARISH"}, enabled=True) is None
    assert run_once._long_btc_regime_veto_reason("LONG", {"btc_regime": "SELF", "trend_4h": "BULL"}, enabled=True) is None
    assert run_once._long_btc_regime_veto_reason("LONG", {"btc_regime": "SELF", "trend_4h": "BEAR"}, enabled=True)

def test_release_version_and_billing_modes_documented():
    # APP_VERSION in the entry point is the single source of truth; a separate
    # VERSION marker is intentionally not required by runtime or CI.
    assert not Path("VERSION").exists()
    runtime = Path("run_once.py").read_text(encoding="utf-8")
    assert 'APP_VERSION = "V10.4.25"' in runtime
    assert 'version=%s' in runtime
    notes = Path('RELEASE_NOTES_V10_4_25.md').read_text(encoding='utf-8')
    assert 'BINGX_ORDER_LINKED_FILLS_AND_INCOME_CONFIRMED' in notes
    assert 'UNKNOWN_DIRECT_FILL_LINK_REQUIRED' in notes
    assert 'estimated_gross_pnl_usdt' in notes
    assert 'active_trades.json' in notes
    assert 'event_engine/reconcile_daily.py' in notes
    tracker = Path('event_engine/tracker.py').read_text(encoding='utf-8')
    assert 'EARLY_LOSS_CUT_ENABLED = False' in tracker
    assert 'EARLY_LOSS_CUT_ENV_REQUESTED' in tracker
    assert 'income_history_unexpected_data_shape' in Path('event_engine/bingx.py').read_text(encoding='utf-8')
    # The current release note documents the current package; obsolete notes are not runtime/test dependencies.
    assert not Path('RELEASE_NOTES_V10_4_4.md').exists()
    assert not Path('RELEASE_NOTES_V10_4_7.md').exists()
    assert not Path('RELEASE_NOTES_V10_4_8.md').exists()
    assert not Path('RELEASE_NOTES_V10_4_9.md').exists()
    assert not Path('RELEASE_NOTES_V10_4_10.md').exists()
    assert not Path('RELEASE_NOTES_V10_4_11.md').exists()
    assert not Path('RELEASE_NOTES_V10_4_12.md').exists()
    assert not Path('RELEASE_NOTES_V10_4_14.md').exists()
    assert not Path('RELEASE_NOTES_V10_4_15.md').exists()


def test_release_runs_bounded_read_only_reconciliation_after_engine():
    workflow = _workflow()
    script = Path("event_engine/reconcile_daily.py").read_text(encoding="utf-8")
    assert "Read-only BingX position and daily ledger reconciliation" in workflow
    assert workflow.index("Run engine") < workflow.index("Read-only BingX position and daily ledger reconciliation")
    assert workflow.index("Stop Binance WireProxy") < workflow.index("Read-only BingX position and daily ledger reconciliation")
    assert workflow.index("Read-only BingX position and daily ledger reconciliation") < workflow.index("Commit state")
    assert 'RECONCILIATION_MAX_SYMBOLS: "2"' in workflow
    assert 'RECONCILIATION_INCOME_REQUEST_BUDGET: "4"' in workflow
    assert "python -m event_engine.reconcile_daily" in workflow
    assert "scripts/reconcile_daily.py" not in workflow
    assert "get_positions" in script and "get_open_orders_snapshot" in script and "get_income_history" in script
    assert "get_open_protection_directional" not in script
    assert "emergency_close_position(" not in script
    assert "place_order(" not in script
    assert "cancel_order(" not in script


def test_early_loss_cut_is_hard_disabled_even_if_environment_requests_true():
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["EARLY_LOSS_CUT_ENABLED"] = "true"
    proc = subprocess.run(
        [sys.executable, "-c", "from event_engine.tracker import EARLY_LOSS_CUT_ENABLED; print(EARLY_LOSS_CUT_ENABLED)"],
        cwd=Path.cwd(), env=env, text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"
    assert "hard-disabled in this release" in proc.stderr
