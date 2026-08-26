from __future__ import annotations

import json
from pathlib import Path


class EventDedup:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.ids: set[str] = set()
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding='utf-8').splitlines():
            try:
                obj = json.loads(line)
                if obj.get('event_id'):
                    self.ids.add(obj['event_id'])
            except Exception:
                continue

    def seen(self, event_id: str) -> bool:
        return event_id in self.ids

    def add(self, event_id: str):
        self.ids.add(event_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({'event_ids': sorted(self.ids)[-10000:]}, indent=2), encoding='utf-8')
