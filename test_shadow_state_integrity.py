import pytest

from event_engine import shadow


@pytest.mark.parametrize("payload", [b'{"trade":BROKEN\n', b'[]', b'{"trade": []}'])
def test_corrupt_shadow_state_is_not_treated_as_empty(tmp_path, payload):
    path = tmp_path / "shadow.json"
    path.write_bytes(payload)

    with pytest.raises(shadow.ShadowStateError):
        shadow._load_state(path)

    assert path.read_bytes() == payload


def test_shadow_update_does_not_overwrite_corrupt_state(tmp_path):
    path = tmp_path / "shadow.json"
    payload = b'{"trade":BROKEN\n'
    path.write_bytes(payload)

    with pytest.raises(shadow.ShadowStateError):
        shadow.update_divergence_shadow_state(path, {"BTC-USDT": 100.0}, 1_800_000_000_000)

    assert path.read_bytes() == payload
