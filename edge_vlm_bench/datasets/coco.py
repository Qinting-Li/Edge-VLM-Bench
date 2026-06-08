from __future__ import annotations

from typing import Any
import json

from edge_vlm_bench.datasets.base import DatasetAdapter


class COCODataset(DatasetAdapter):
    task = "detect"

    def __init__(self, config: dict[str, Any], demo: bool = False):
        super().__init__(config, demo)
        self.images, self.annotations = self._load()

    def _load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        annotation_file = self.resolve_path(self.config.get("annotation_file"))
        if annotation_file and annotation_file.exists():
            payload = json.loads(annotation_file.read_text(encoding="utf-8"))
            return payload.get("images", []), payload.get("annotations", [])
        if self.demo:
            images = [{"id": 1, "file_name": "demo.jpg", "width": 640, "height": 640}]
            annotations = [
                {"image_id": 1, "bbox": [10, 10, 100, 100], "category_id": 1, "iscrowd": 0}
            ]
            return images, annotations
        raise FileNotFoundError("COCO dataset requires dataset.annotation_file unless --demo is used")

    def get_batch(self, index: int, batch_size: int) -> list[dict[str, Any]]:
        start = (index * batch_size) % len(self.images)
        batch = []
        for offset in range(batch_size):
            image = self.images[(start + offset) % len(self.images)]
            batch.append(
                {
                    "image_id": image.get("id", start + offset),
                    "path": image.get("file_name"),
                    "category_id": self._category_for_image(image.get("id")),
                }
            )
        return batch

    def ground_truth(self) -> list[dict[str, Any]]:
        return self.annotations

    def _category_for_image(self, image_id: int | None) -> int:
        for ann in self.annotations:
            if ann.get("image_id") == image_id:
                return int(ann.get("category_id", 1))
        return 1

