from __future__ import annotations

from typing import Any
import random
import time

from edge_vlm_bench.models.base import ModelAdapter


class DummyAdapter(ModelAdapter):
    family = "dummy"

    def load(self) -> None:
        time.sleep(float(self.config.get("load_sleep_s", 0.01)))

    def preprocess(self, batch: list[Any]) -> list[Any]:
        time.sleep(float(self.config.get("preprocess_sleep_s", 0.001)))
        return batch

    def infer(self, prepared: list[Any]) -> list[dict[str, Any]]:
        time.sleep(float(self.config.get("infer_sleep_s", 0.005)))
        rng = random.Random(7)
        return [
            {
                "image_id": item.get("image_id", idx),
                "bbox": [10, 10, 100, 100],
                "score": 0.9,
                "category_id": item.get("category_id", 1),
                "mask_iou": 0.8 + rng.random() * 0.05,
                "answer": "demo",
            }
            for idx, item in enumerate(prepared)
        ]

    def postprocess(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        time.sleep(float(self.config.get("postprocess_sleep_s", 0.001)))
        return raw

