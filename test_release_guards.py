from pathlib import Path


def test_clean_release_uses_divergence_shadow_mode():
    workflow = Path('.github/workflows/event-engine.yml').read_text(encoding='utf-8')
    assert 'DIVERGENCE_SHADOW_ONLY: "true"' in workflow


def test_clean_release_has_pre_order_drift_retry_budget():
    workflow = Path('.github/workflows/event-engine.yml').read_text(encoding='utf-8')
    assert 'MAX_PRE_ORDER_DRIFT_REJECTIONS: "3"' in workflow
