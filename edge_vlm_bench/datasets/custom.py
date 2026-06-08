from __future__ import annotations

from typing import Any
import json

from edge_vlm_bench.datasets.base import DatasetAdapter


class CustomDataset(DatasetAdapter):
    task = "custom"

    def __init__(self, config: dict[str, Any], demo: bool = False):
        super().__init__(config, demo)
        self.items = self._load_items()

    def _load_items(self) -> list[dict[str, Any]]:
        manifest = self.resolve_path(self.config.get("manifest"))
        if manifest and manifest.exists():
            return json.loads(manifest.read_text(encoding="utf-8"))
        if self.demo:
            return [
                {"image_id": 1, "path": "demo/image_001.jpg", "category_id": 1},
                {"image_id": 2, "path": "demo/image_002.jpg", "category_id": 1},
            ]
        raise FileNotFoundError("custom dataset requires dataset.manifest unless --demo is used")

    def get_batch(self, index: int, batch_size: int) -> list[dict[str, Any]]:
        start = (index * batch_size) % len(self.items)
        return [self.items[(start + offset) % len(self.items)] for offset in range(batch_size)]

    def ground_truth(self) -> list[dict[str, Any]]:
        return self.items

