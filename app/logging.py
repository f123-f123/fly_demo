from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import ROOT


class SessionLogger:
    def __init__(self, logs_dir: Path = ROOT / "logs", max_bytes: int = 20_000_000):
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = logs_dir / f"session-{stamp}.jsonl"
        self.max_bytes = max_bytes
        self.capped = False
        self._file = self.path.open("w", encoding="utf-8")

    def write(self, state: dict[str, object]) -> None:
        if self.capped:
            return
        line = json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n"
        if self._file.tell() + len(line.encode("utf-8")) > self.max_bytes:
            self.capped = True
            self._file.flush()
            return
        self._file.write(line)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
