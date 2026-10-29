"""Release guard: event engines active in archived V12 must remain enabled.

This intentionally checks the workflow configuration, not exchange execution.
Shadow event flags are telemetry and do not veto orders while ENTRY_QUALITY_MODE=shadow.
"""
from pathlib import Path
import re

WORKFLOW = Path(".github/workflows/event-engine.yml")


def _env() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in WORKFLOW.read_text(encoding="utf-8").splitlines():
        match = re.match(r'^\s{6}([A-Z][A-Z0-9_]+):\s*(.*?)\s*$', line)
        if match:
            value = match.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            result[match.group(1)] = value
    return result


def test_all_pattern_engines_enabled_in_v12_remain_enabled_in_v10418():
    env = _env()
    enabled_in_v12 = {
        "ENABLE_DIVERGENCE_ENGINE",
        "ENABLE_VOLUME_PROFILE_DIVERGENCE_ENGINE",
        "ENABLE_HARMONIC_PATTERN_ENGINE",
        "ENABLE_MACD_4H_ENGINE",
        "ENABLE_MA_COMPRESSION_ENGINE",
        "ENABLE_BREAKOUT_MOMENTUM_ENGINE",
        "ENABLE_DONCHIAN_RETEST_ENGINE",
        "ENABLE_LIQUIDITY_SWEEP_ENGINE",
        "ENABLE_EMA_PULLBACK_ENGINE",
        "ENABLE_ORDER_BLOCK_ENGINE",
        "ENABLE_BREAKER_BLOCK_ENGINE",
        "ENABLE_SFP_ENGINE",
        "ENABLE_CRT_ENGINE",
    }
    for flag in sorted(enabled_in_v12):
        assert env.get(flag, "false").lower() == "true", f"V12-enabled pattern engine disabled: {flag}"


def test_engines_disabled_in_v12_are_not_accidentally_enabled():
    env = _env()
    disabled_in_v12 = {
        "ENABLE_MITIGATION_BLOCK_ENGINE",
        "ENABLE_MITIGATION_BLOCK_BULLISH_ENGINE",
        "ENABLE_LIQUIDATION_CASCADE_FVG_ENGINE",
        "LIQ_SQUEEZE_ENGINE_ENABLED",
    }
    for flag in sorted(disabled_in_v12):
        assert env.get(flag, "false").lower() == "false", f"Pattern engine disabled in V12 is unexpectedly enabled: {flag}"


def test_pattern_shadow_flags_do_not_hard_block_events():
    env = _env()
    assert env.get("ENTRY_QUALITY_MODE") == "shadow"
    assert env.get("ENTRY_WEAK_ENGINE_BLOCK_ENABLED", "false").lower() == "true"
    assert env.get("DISABLED_EVENT_TYPES") == ""
    shadow_flags = set(filter(None, (x.strip() for x in env.get("ENTRY_BLOCKED_EVENT_TYPES", "").split(","))))
    assert shadow_flags == {
        "CRT_BULLISH",
        "CRT_BEARISH",
        "MA_COMPRESSION_BREAKOUT",
        "EMA_PULLBACK_CONTINUATION",
        "LIQUIDITY_SWEEP_RECLAIM",
        "VOLUME_PROFILE_DISTRIBUTION",
        "BREAKER_BLOCK_BEARISH",
    }


def test_breaker_bullish_and_breakout_momentum_are_not_hard_denied():
    env = _env()
    denylist = {x.strip().upper() for x in env.get("DISABLED_EVENT_TYPES", "").split(",") if x.strip()}
    assert "BREAKOUT_MOMENTUM" not in denylist
    assert "BREAKER_BLOCK_BULLISH" not in denylist
    assert "CRT_BULLISH" not in denylist
    assert env.get("ENABLE_BREAKOUT_MOMENTUM_ENGINE", "false").lower() == "true"
    assert env.get("ENABLE_BREAKER_BLOCK_ENGINE", "false").lower() == "true"
