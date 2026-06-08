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


class TorchDummyAdapter(DummyAdapter):
    family = "torch_dummy"

    def load(self) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(self.config.get("device", "cuda:0"))
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        size = int(self.config.get("matrix_size", 1024))
        self.weights = torch.randn(size, size, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def infer(self, prepared: list[Any]) -> list[dict[str, Any]]:
        size = self.weights.shape[0]
        x = self.torch.randn(size, size, device=self.device)
        y = x @ self.weights
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        predictions = super().infer(prepared)
        for prediction in predictions:
            prediction["backend_device"] = str(self.device)
            prediction["compute_checksum"] = float(y[0, 0].detach().cpu())
        return predictions

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data["torch_device"] = str(getattr(self, "device", self.config.get("device", "unknown")))
        if hasattr(self, "torch") and self.device.type == "cuda":
            data["torch_cuda_name"] = self.torch.cuda.get_device_name(self.device)
        return data
