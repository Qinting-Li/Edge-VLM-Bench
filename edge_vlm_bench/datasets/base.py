from __future__ import annotations

from pathlib import Path
from typing import Any


class DatasetAdapter:
    task = "detect"

    def __init__(self, config: dict[str, Any], demo: bool = False):
        self.config = config
        self.demo = demo

    def sample_batch(self, batch_size: int) -> list[dict[str, Any]]:
        return self.get_batch(0, batch_size)

    def get_batch(self, index: int, batch_size: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def ground_truth(self) -> list[dict[str, Any]]:
        return []

    @staticmethod
    def resolve_path(value: str | None) -> Path | None:
        return Path(value).expanduser().resolve() if value else None

