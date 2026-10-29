from __future__ import annotations

import gzip
import json

import run_once as ro


def test_record_action_rotates_under_threshold_and_reads_complete_history(tmp_path, monkeypatch):
    journal = tmp_path / "data" / "actions.jsonl"
    monkeypatch.setattr(ro, "ACTIONS", journal)
    monkeypatch.setattr(ro, "ACTION_JOURNAL_ROTATE_BYTES", 320)

    expected = []
    for i in range(17):
        row = {"record_type": "TEST", "sequence": i, "detail": "x" * 46}
        expected.append(row)
        ro.record_action(row)

    actual = list(ro._iter_action_records())
    assert actual == expected
    assert journal.stat().st_size <= ro.ACTION_JOURNAL_ROTATE_BYTES
    archives = sorted((journal.parent / "archive").glob("actions-*.jsonl.gz"))
    assert len(archives) >= 2

    # Every compressed archive independently contains valid JSONL rows.
    archived_rows = []
    for archive in archives:
        with gzip.open(archive, "rt", encoding="utf-8") as stream:
            for line in stream:
                archived_rows.append(json.loads(line))
    assert archived_rows + [r for r in actual if r not in archived_rows] == expected


def test_interrupted_rotation_duplicate_is_not_read_twice_and_is_recovered(tmp_path, monkeypatch):
    journal = tmp_path / "data" / "actions.jsonl"
    archive_dir = journal.parent / "archive"
    archive_dir.mkdir(parents=True)
    original_rows = [{"sequence": i} for i in range(8)]
    raw = "".join(json.dumps(row) + "\n" for row in original_rows).encode("utf-8")
    journal.write_bytes(raw)
    archived = archive_dir / "actions-20991231T235959.999999Z.jsonl.gz"
    with gzip.open(archived, "wb") as stream:
        stream.write(raw)

    monkeypatch.setattr(ro, "ACTIONS", journal)
    monkeypatch.setattr(ro, "ACTION_JOURNAL_ROTATE_BYTES", len(raw))

    # Simulates a stop after archive installation but before active-journal reset.
    assert list(ro._iter_action_records()) == original_rows
    ro.record_action({"sequence": 8})
    assert list(ro._iter_action_records()) == original_rows + [{"sequence": 8}]
    assert journal.read_text(encoding="utf-8") == json.dumps({"sequence": 8}) + "\n"
    assert len(list(archive_dir.glob("actions-*.jsonl.gz"))) == 1


def test_action_reader_skips_a_truncated_line_but_reads_following_rows(tmp_path, monkeypatch):
    journal = tmp_path / "data" / "actions.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_bytes(b'{"sequence":1}\n{"sequence":BROKEN\n{"sequence":3}\n')
    monkeypatch.setattr(ro, "ACTIONS", journal)

    assert list(ro._iter_action_records()) == [{"sequence": 1}, {"sequence": 3}]


def test_startup_rotation_uses_same_verified_helper(tmp_path, monkeypatch):
    journal = tmp_path / "data" / "actions.jsonl"
    journal.parent.mkdir(parents=True)
    expected = b'{"sequence":1}\n' * 12
    journal.write_bytes(expected)
    monkeypatch.setattr(ro, "ACTIONS", journal)
    monkeypatch.setattr(ro, "ACTION_JOURNAL_ROTATE_BYTES", 100)

    archive = ro._ensure_action_journal_safe()

    assert archive is not None and archive.exists()
    with gzip.open(archive, "rb") as stream:
        assert stream.read() == expected
    assert journal.read_bytes() == b""
